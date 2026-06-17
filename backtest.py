"""
NSE Momentum Loss Screener — Walk-Forward Backtesting Engine
============================================================
Simulates the 3-gate CE sell strategy over historical daily prices.

Gate 1 : Stock surges ≥ min_gain_percent continuously (green candles /
          red candles allowed only if they hold higher-low structure).
          Followed by a breakdown candle: close < prev_low with volume
          ≥ ratio × 20d avg.

Gate 2 : After the breakdown, price closes below 9-EMA at least once
          with above-average volume  (sticky confirmation).

Gate 3 : Price rallies back within price_proximity_percent of surge_high
          → Sell a slightly OTM CE above surge_high.
          Trade held until that month's NSE expiry (last Thursday).

Outcome :
    WIN  → stock closes below strike at expiry   (CE expires worthless)
    LOSS → stock closes ≥ strike at expiry       (CE ITM / assignment risk)

Run standalone:
    python backtest.py --tickers RELIANCE TCS HDFCBANK --years 2

As FastAPI router, mount in main.py:
    from backtest import router as bt_router
    app.include_router(bt_router)
"""

import argparse
import hashlib
import json
import math
import sys
from calendar import monthrange
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from scipy.stats import norm


# ── Optional FastAPI import (not required for standalone use) ──────────────
try:
    from fastapi import APIRouter, BackgroundTasks
    from fastapi.responses import JSONResponse
    _has_fastapi = True
except ImportError:
    _has_fastapi = False

# ── Optional Fair Value import (from sma_router) ──────────────────────────
# Fair value enriches every Gate 3 trade with regression-based fundamental
# data (Operating Profit, Sales, TTM models scraped from Screener.in).
# If sma_router is unavailable the backtest runs normally without FV data.
try:
    from sma_router import _analyze_ticker as _sma_analyze
    _has_fair_value = True
except ImportError:
    _has_fair_value = False

# In-process fair value cache: ticker → {composite_fair_price, composite_gain_pct,
# valuation_bucket, model_count, current_price, op_fair, sales_fair, ttm_fair}
# Populated lazily on first Gate 3 hit; reused for all trades of the same ticker.
_fv_cache: Dict[str, Dict] = {}


def _get_fair_value(ticker: str, force: bool = False) -> Dict:
    """
    Fetch fundamental fair value for ticker using SMA regression models.
    Returns a flat dict with fair value fields, or an empty dict on failure.
    Results are cached in-process so each ticker is only scraped once per run.
    """
    if not _has_fair_value:
        return {}

    global _fv_cache
    cache_key = ticker.upper()

    if not force and cache_key in _fv_cache:
        return _fv_cache[cache_key]

    try:
        result = _sma_analyze(
            ticker=ticker,
            fy_start=2014,
            force=False,
            include_other_income=True,
        )
        if "error" in result:
            out = {"fv_error": result["error"]}
        else:
            op    = result.get("op",  {})
            sales = result.get("sales", {})
            ttm   = result.get("ttm",  {})
            out = {
                "fv_current_price":      result.get("current_price"),
                "fv_composite_fair":     result.get("composite_fair_price"),
                "fv_composite_gain_pct": result.get("composite_gain_pct"),
                "fv_valuation_bucket":   result.get("valuation_bucket"),  # UNDERVALUED / FAIR / OVERVALUED
                "fv_model_count":        result.get("model_count", 0),
                # Individual model fair prices (for tooltip / drill-down)
                "fv_op_fair":            op.get("pred_price"),
                "fv_op_r2":             op.get("r2"),
                "fv_op_gain_pct":       op.get("gain_pct"),
                "fv_sales_fair":         sales.get("pred_price") if result.get("has_sales") else None,
                "fv_sales_r2":          sales.get("r2")         if result.get("has_sales") else None,
                "fv_ttm_fair":           ttm.get("pred_price")  if result.get("has_ttm")   else None,
                "fv_ttm_r2":            ttm.get("r2")          if result.get("has_ttm")    else None,
            }
        _fv_cache[cache_key] = out
        return out
    except Exception as e:
        out = {"fv_error": str(e)}
        _fv_cache[cache_key] = out
        return out

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
IST   = pytz.timezone("Asia/Kolkata")
RISK_FREE_RATE = 0.065      # India approximate 10Y G-Sec yield
VOL_WINDOW     = 30         # days for historical vol estimate
MIN_WARMUP     = 60         # trading days needed before first signal scan

DEFAULT_CONFIG = {
    "lookback_days":              252,    # 1 trading year history per scan window
    "min_gain_percent":           18.0,
    "min_green_candles":          2,
    "surge_recency_days":         45,
    "min_drop_percent":           0.1,
    "min_breakdown_volume_ratio": 0.5,
    "ema_period":                 9,
    "price_proximity_percent":    1.0,
    "sell_zone_lookback_days":    10,
    "max_gate3_days":              90,     # calendar days after Gate 2 to scan for Gate 3 retest; 0 = unlimited
    # Expiry control ─────────────────────────────────────────────────────────
    # "auto"    → current month; roll to next if days_to_expiry < min_days_to_expiry
    # "current" → always current month (even if very close)
    # "next"    → always next month expiry
    "expiry_mode":          "auto",
    "min_days_to_expiry":   5,       # roll threshold (auto mode only)
    # Premium Surge Analysis
    # At Gate 3 retest, vol_surge_at_g3 = volume / 20d avg volume.
    # Trades where this >= threshold are flagged as "High Premium" setups.
    "premium_vol_surge_threshold": 1.5,
    # ── Max OTM Gain Pact ────────────────────────────────────────────────────
    # Instead of always selling the +1 OTM strike, evaluate every strike from
    # +1 to +max_otm_depth OTM and pick the one with the highest Expected Value:
    #   EV = avg_premium × win_rate_at_that_offset
    # The backtest records per-offset results in strike_variants[] and builds
    # an otm_bucket_stats table in the summary for each ticker.
    # The live screener (main.py) reads this table to auto-select the optimal
    # strike when max_otm_ev_mode is True.
    "max_otm_depth":              3,     # evaluate +1, +2, +3 OTM
    "max_otm_ev_mode":            True,  # kept for compat; use otm_selection_mode instead
    # otm_selection_mode controls which OTM offset is used as the PRIMARY trade:
    #   "safest"  → deepest available OTM with premium >= min_tradable (max safety)
    #   "ev"      → EV-optimal offset (premium × win_rate, from two-pass backtest)
    #   "default" → always +1 OTM (original behaviour)
    "otm_selection_mode":         "safest",   # ← safest = highest OTM = safer trade
    "min_tradable_premium":       0.10,        # Rs — stop building variants below this
}

# ─────────────────────────────────────────────────────────────────────────────
#  DISK CACHE
#  Results are keyed by MD5( sorted_tickers + years + full_config ).
#  Cache lives in ./backtest_cache/<hash>.json beside this file.
#  A cached entry is reused unless the caller passes force=True.
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent / "backtest_cache"


def _cache_key(tickers: List[str], years: int, cfg: Dict) -> str:
    """Stable MD5 key for a (tickers, years, config) combination."""
    payload = json.dumps(
        {
            "tickers": sorted(t.upper() for t in tickers),
            "years":   years,
            "config":  {k: cfg[k] for k in sorted(cfg)},
        },
        sort_keys=True,
    )
    return hashlib.md5(payload.encode()).hexdigest()


def _cache_load(key: str) -> Optional[Dict]:
    """Return cached result dict or None if not found."""
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass   # corrupt file — treat as cache miss
    return None


def _cache_save(key: str, data: Dict) -> None:
    """Persist result dict to disk under the given key."""
    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"{key}.json"
        with open(path, "w") as f:
            json.dump(data, f, separators=(",", ":"))
        print(f"[cache] saved → {path.name}")
    except Exception as e:
        print(f"[cache] save failed: {e}", file=sys.stderr)


def _cache_list() -> List[Dict]:
    """Return metadata for every cached entry (for the /cache endpoint)."""
    if not CACHE_DIR.exists():
        return []
    entries = []
    for p in sorted(CACHE_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(p) as f:
                d = json.load(f)
            agg = d.get("aggregate", {})
            entries.append({
                "key":       p.stem,
                "tickers":   agg.get("tickers", []),
                "years":     d.get("years_tested"),
                "trades":    agg.get("total_trades", 0),
                "accuracy":  agg.get("accuracy_pct", 0),
                "cached_at": d.get("cached_at", ""),
                "size_kb":   round(p.stat().st_size / 1024, 1),
            })
        except Exception:
            pass
    return entries


def _cache_delete(key: str) -> bool:
    """Delete a single cache entry. Returns True if deleted."""
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        path.unlink()
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
#  OPTION MATHS
# ─────────────────────────────────────────────────────────────────────────────
def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black-Scholes European call price. T in years."""
    if T <= 0:
        return max(0.0, S - K)
    if sigma <= 0:
        return max(0.0, S - K)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


def hist_vol(closes: np.ndarray, window: int = VOL_WINDOW) -> float:
    """Annualised historical volatility from daily close prices."""
    if len(closes) < window + 1:
        window = len(closes) - 1
    if window < 2:
        return 0.30   # fallback 30%
    log_ret = np.log(closes[-(window + 1):][1:] / closes[-(window + 1):][:-1])
    return float(np.std(log_ret, ddof=1) * math.sqrt(252))


# ─────────────────────────────────────────────────────────────────────────────
#  NSE UTILITIES
# ─────────────────────────────────────────────────────────────────────────────
def nse_monthly_expiry(year: int, month: int) -> date:
    """Last Thursday of the given month (NSE monthly F&O expiry)."""
    _, last_day_num = monthrange(year, month)
    d = date(year, month, last_day_num)
    while d.weekday() != 3:   # 3 = Thursday
        d -= timedelta(days=1)
    return d


def nse_strike_interval(price: float) -> float:
    """NSE standard strike interval for a given stock price."""
    if price < 250:
        return 10
    if price < 500:
        return 20
    if price < 1000:
        return 50
    if price < 2500:
        return 100
    if price < 5000:
        return 200
    return 500


def nearest_otm_strike(surge_high: float) -> float:
    """
    Nearest strike ABOVE surge_high, rounded to NSE standard interval.
    'Slightly OTM' = closest available strike above the resistance level.
    """
    interval = nse_strike_interval(surge_high)
    return math.ceil(surge_high / interval) * interval


# ─────────────────────────────────────────────────────────────────────────────
#  DATA FETCHING
# ─────────────────────────────────────────────────────────────────────────────
def fetch_history(ticker: str, years: int = 2) -> Optional[pd.DataFrame]:
    """Fetch `years` of daily OHLCV for `ticker` (NSE) via yfinance."""
    try:
        end   = datetime.now(IST).date() + timedelta(days=1)
        start = end - timedelta(days=years * 366)
        df = yf.Ticker(f"{ticker}.NS").history(
            start=str(start), end=str(end), auto_adjust=False
        )
        if df.empty:
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert(IST)
        df = df.sort_index()
        # auto_adjust=False may return MultiIndex columns on newer yfinance versions;
        # flatten to simple column names so downstream code works unchanged.
        if isinstance(df.columns, type(df.columns)) and hasattr(df.columns, 'get_level_values'):
            try:
                df.columns = df.columns.get_level_values(0)
            except Exception:
                pass
        # Keep only the OHLCV columns we need (drops Adj Close, Dividends, etc.)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            if col not in df.columns:
                print(f"[yf] {ticker}: missing column {col}", file=sys.stderr)
                return None
        return df[["Open", "High", "Low", "Close", "Volume"]]
    except Exception as e:
        print(f"[yf] {ticker}: {e}", file=sys.stderr)
        return None


# ─────────────────────────────────────────────────────────────────────────────
#  CORE SIGNAL LOGIC  (ported from main.py — no live price injection)
# ─────────────────────────────────────────────────────────────────────────────
def compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    return pd.Series(values, dtype=float).ewm(span=period, adjust=False).mean().values


def check_surge_continuity(
    closes: np.ndarray, opens: np.ndarray, lows: np.ndarray,
    start: int, end: int
) -> Tuple[bool, int, int]:
    green_count = allowed_red = 0
    for i in range(start, end + 1):
        if closes[i] > opens[i]:
            green_count += 1
        else:
            if i > start and closes[i] < lows[i - 1]:
                return False, allowed_red, green_count
            allowed_red += 1
    return True, allowed_red, green_count


def detect_breakdown(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    opens: np.ndarray, vols: np.ndarray,
    idx: int, cfg: Dict
) -> Optional[Dict]:
    """
    Check if candle at `idx` is a valid breakdown candle.
    Returns breakdown info dict or None.
    Looks back up to surge_recency_days to find breakdown, then checks surge.
    """
    min_drop      = cfg.get("min_drop_percent", 0.1)
    min_vol_ratio = cfg.get("min_breakdown_volume_ratio", 0.5)
    lookback_bd   = int(cfg.get("surge_recency_days", 45))

    # ── Step 1: Find a valid breakdown candle within lookback window ──────────
    breakdown_idx = None
    yesterday_high = yesterday_low = drop_pct = breakdown_vol = None
    avg_vol_20d = volume_ratio = ema_val = None

    for offset in range(0, min(lookback_bd + 1, idx)):
        bd_idx   = idx - offset
        prev_idx = bd_idx - 1
        if prev_idx < 1:
            break

        bd_close  = float(closes[bd_idx])
        prev_low  = float(lows[prev_idx])
        prev_high = float(highs[prev_idx])

        if bd_close >= prev_low:
            continue
        _drop = (prev_low - bd_close) / prev_low * 100
        if _drop < min_drop:
            continue

        _bd_vol  = float(vols[bd_idx]) if vols[bd_idx] > 0 else 0.0
        _vol_win = vols[max(0, bd_idx - 20):bd_idx]
        _avg_vol = float(_vol_win.mean()) if len(_vol_win) > 0 else 0.0
        _v_ratio = (_bd_vol / _avg_vol) if _avg_vol > 0 else 0.0
        if _v_ratio < min_vol_ratio:
            continue

        breakdown_idx  = bd_idx
        yesterday_high = prev_high
        yesterday_low  = prev_low
        drop_pct       = _drop
        breakdown_vol  = _bd_vol
        avg_vol_20d    = _avg_vol
        volume_ratio   = _v_ratio
        ema_val        = None   # computed below
        break

    if breakdown_idx is None:
        return None

    # ── Step 2: Verify continuous surge ending just before breakdown ──────────
    min_gain  = cfg.get("min_gain_percent", 18.0)
    min_green = cfg.get("min_green_candles", 2)
    recency   = int(cfg.get("surge_recency_days", 45))
    ema_period= int(cfg.get("ema_period", 9))

    scan_closes = closes[:breakdown_idx]
    scan_opens  = opens[:breakdown_idx]
    scan_lows   = lows[:breakdown_idx]
    scan_highs  = highs[:breakdown_idx]
    n = len(scan_closes)
    if n < min_green + 2:
        return None

    min_end_idx  = max(0, n - recency)
    window_min   = max(min_green + 1, 3)
    best_gain    = 0.0
    best_window  = None
    best_greens  = 0

    for wsize in range(window_min, n + 1):
        for start in range(0, n - wsize + 1):
            end_ = start + wsize - 1
            if end_ < min_end_idx:
                continue
            net_gain = (scan_closes[end_] - scan_closes[start]) / scan_closes[start] * 100
            if net_gain < min_gain:
                continue
            is_cont, _, green_count = check_surge_continuity(
                scan_closes, scan_opens, scan_lows, start, end_
            )
            if not is_cont or green_count < min_green:
                continue
            prev_end = best_window[1] if best_window else -1
            if net_gain > best_gain or (net_gain == best_gain and end_ > prev_end):
                best_gain   = net_gain
                best_greens = green_count
                best_window = (start, end_)

    if best_window is None or best_gain < min_gain:
        return None

    surge_start_idx, surge_end_idx = best_window
    surge_high = float(np.max(scan_highs[surge_start_idx:surge_end_idx + 1]))

    # EMA at breakdown
    ema_vals  = compute_ema(closes[:breakdown_idx + 1], ema_period)
    ema_at_bd = float(ema_vals[-1])

    return {
        "breakdown_idx":  breakdown_idx,
        "surge_high":     round(surge_high, 2),
        "yesterday_high": round(float(yesterday_high), 2),
        "yesterday_low":  round(float(yesterday_low), 2),
        "drop_pct":       round(drop_pct, 2),
        "volume_ratio":   round(volume_ratio, 2),
        "surge_gain_pct": round(best_gain, 2),
        "surge_candles":  best_greens,
        "surge_start_idx":surge_start_idx,
        "surge_end_idx":  surge_end_idx,
        "ema_at_breakdown": round(ema_at_bd, 2),
    }


def check_gate2(
    closes: np.ndarray, vols: np.ndarray,
    bd_idx: int, cfg: Dict
) -> Optional[int]:
    """
    Scan from bd_idx+1 onward (candle AFTER breakdown). Return first index where
    close < 9-EMA AND volume ≥ min_breakdown_volume_ratio × 20d avg.
    """
    ema_period    = int(cfg.get("ema_period", 9))
    min_vol_ratio = float(cfg.get("min_breakdown_volume_ratio", 0.5))
    ema_vals      = compute_ema(closes, ema_period)
    avg_vol_20d   = float(np.mean(vols[-20:])) if len(vols) >= 20 else float(np.mean(vols))
    if avg_vol_20d <= 0:
        return None
    for i in range(bd_idx + 1, len(closes)):   # +1: skip breakdown candle itself
        if closes[i] < ema_vals[i]:
            if float(vols[i]) / avg_vol_20d >= min_vol_ratio:
                return i
    return None


def check_gate3(
    closes: np.ndarray, highs: np.ndarray,
    g2_idx: int, surge_high: float, cfg: Dict,
    dates: object = None,   # pd.DatetimeIndex — used for max_gate3_days limit
) -> Optional[int]:
    prox_pct       = float(cfg.get("price_proximity_percent", 1.0))
    max_g3_days    = int(cfg.get("max_gate3_days", 90))   # 0 = unlimited
    floor          = surge_high * (1 - prox_pct / 100)

    # Calendar-day deadline: if Gate 3 doesn't fire within max_gate3_days
    # after Gate 2, the setup is stale (e.g. stock already in a new uptrend).
    deadline_date  = None
    if max_g3_days > 0 and dates is not None:
        from datetime import timedelta
        deadline_date = dates[g2_idx].date() + timedelta(days=max_g3_days)

    pulled_back       = False   # True once close has dropped below floor after Gate 2
    above_high_streak = 0       # consecutive closes above surge_high

    # Start from g2_idx+1: Gate 2 candle itself is never a valid entry.
    # Breakout requires 2 consecutive closes above surge_high — a single spike
    # candle (high above surge_high but close well below, e.g. Jan 31 2022 AUBANK)
    # should NOT permanently kill Gate 3 scanning.
    for i in range(g2_idx + 1, len(closes)):
        close = float(closes[i])
        high  = float(highs[i])

        # Stale-setup guard: abort if past max_gate3_days deadline
        if deadline_date is not None and dates[i].date() > deadline_date:
            return None

        # Track consecutive closes above surge_high (confirmed breakout)
        if close > surge_high:
            above_high_streak += 1
            if above_high_streak >= 2:
                return None   # 2 consecutive closes above → thesis dead
            # Single close above: don't fire entry, but keep scanning
            pulled_back = False  # reset pullback — price escaped zone
            continue
        else:
            above_high_streak = 0  # streak broken

        # Phase 1: mark pullback when close falls below the floor
        if not pulled_back and close < floor:
            pulled_back = True

        # Phase 2: retest — HIGH re-enters the sell zone AFTER confirmed pullback
        elif pulled_back and high >= floor:
            return i

    return None


# ─────────────────────────────────────────────────────────────────────────────
#  TRADE EVALUATION
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_trade(
    closes: np.ndarray, dates: pd.DatetimeIndex,
    g3_idx: int, surge_high: float, cfg: Dict,
    vols: Optional[np.ndarray] = None,
    recommended_otm_idx: int = 1,
) -> Optional[Dict]:
    """
    Max OTM Gain Pact primary flow
    --------------------------------
    1. Evaluate ALL strikes +1 to +max_otm_depth OTM (strike_variants[]).
       Stop when Black-Scholes premium < Rs 0.10 (not tradable).
    2. PRIMARY trade uses recommended_otm_idx (from EV table of prior run).
       First run defaults to +1 OTM; subsequent runs auto-use EV-best offset.
    3. BCS (Bear Call Spread) computed for every variant and the primary trade.
    4. Single-close breakout kill is REMOVED (handled in check_gate3 with
       2-consecutive-close rule so spike candles like Jan31 2022 are correctly
       caught as retests, not false breakouts).
    """
    entry_price = float(closes[g3_idx])
    entry_date  = dates[g3_idx].date()

    interval    = nse_strike_interval(surge_high)
    base_strike = nearest_otm_strike(surge_high)

    expiry_mode   = str(cfg.get("expiry_mode", "auto")).lower()
    min_days_roll = int(cfg.get("min_days_to_expiry", 5))

    def _next_expiry(d):
        nm = d.month + 1 if d.month < 12 else 1
        ny = d.year + (1 if d.month == 12 else 0)
        return nse_monthly_expiry(ny, nm)

    if expiry_mode == "next":
        expiry_date = _next_expiry(entry_date)
    elif expiry_mode == "current":
        expiry_date = nse_monthly_expiry(entry_date.year, entry_date.month)
    else:
        expiry_date = nse_monthly_expiry(entry_date.year, entry_date.month)
        if (expiry_date - entry_date).days <= min_days_roll:
            expiry_date = _next_expiry(entry_date)

    T_years = max(0.001, (expiry_date - entry_date).days / 365.0)
    sigma   = hist_vol(closes[:g3_idx + 1], VOL_WINDOW)

    exp_dates = [d.date() for d in dates]
    exp_idx   = None
    for search_offset in range(0, 5):
        target = expiry_date + timedelta(days=search_offset)
        if target in exp_dates:
            exp_idx = exp_dates.index(target)
            break
    if exp_idx is None or exp_idx >= len(closes):
        return None

    expiry_close  = float(closes[exp_idx])
    actual_expiry = dates[exp_idx].date()

    # Build all OTM strike variants with BCS
    max_otm_depth   = int(cfg.get("max_otm_depth", 3))
    strike_variants: List[Dict] = []
    for otm_idx in range(1, max_otm_depth + 1):
        k    = base_strike + (otm_idx - 1) * interval
        prem = round(bs_call(entry_price, k, T_years, RISK_FREE_RATE, sigma), 2)
        if prem < 0.10:
            break
        intrin  = max(0.0, expiry_close - k)
        pnl_raw = prem - intrin
        pnl_k   = max(pnl_raw, -3 * prem)
        hedge_k    = k + interval
        hedge_prem = round(bs_call(entry_price, hedge_k, T_years, RISK_FREE_RATE, sigma), 2)
        bcs_net    = round(prem - hedge_prem, 2) if hedge_prem > 0 else None
        bcs_ml     = round(interval - bcs_net, 2) if (bcs_net and bcs_net > 0) else None
        bcs_be     = round(k + bcs_net, 2) if bcs_net else None
        bcs_rr     = round(bcs_ml / bcs_net, 2) if (bcs_net and bcs_net > 0 and bcs_ml) else None
        strike_variants.append({
            "otm_idx":          otm_idx,
            "strike":           int(k),
            "premium":          prem,
            "result":           "WIN" if expiry_close < k else "LOSS",
            "pnl":              round(pnl_k, 2),
            "return_pct":       round(pnl_raw / prem * 100, 1) if prem > 0 else 0.0,
            "bcs_hedge_strike": int(hedge_k),
            "bcs_hedge_prem":   hedge_prem,
            "bcs_net_premium":  bcs_net,
            "bcs_max_profit":   bcs_net,
            "bcs_max_loss":     bcs_ml,
            "bcs_breakeven":    bcs_be,
            "bcs_risk_reward":  bcs_rr,
        })

    if not strike_variants:
        return None

    # ── Select PRIMARY strike based on otm_selection_mode ────────────────────
    #
    #   "safest"  → deepest available OTM (highest otm_idx with tradable premium)
    #               Furthest from current price = lowest probability of going ITM
    #               Lower premium collected but much higher win rate = safer trade
    #
    #   "ev"      → EV-optimal: offset with best (premium × win_rate) from
    #               two-pass backtest. recommended_otm_idx is passed in by
    #               backtest_multiple after Pass 1 builds the EV table.
    #
    #   "default" → always +1 OTM (nearest strike above surge_high)
    #
    sel_mode = str(cfg.get("otm_selection_mode", "safest")).lower()
    avail    = {v["otm_idx"]: v for v in strike_variants}

    if sel_mode == "safest":
        # Deepest variant = last in strike_variants list (highest otm_idx)
        primary_v = strike_variants[-1]
    elif sel_mode == "ev":
        # Use recommended offset from EV table; fall back to deepest if unavailable
        primary_v = avail.get(recommended_otm_idx, strike_variants[-1])
    else:
        # "default" — always +1 OTM
        primary_v = strike_variants[0]

    strike           = primary_v["strike"]
    entry_ce_premium = primary_v["premium"]
    result           = primary_v["result"]
    pnl              = primary_v["pnl"]
    intrinsic_at_expiry = max(0.0, expiry_close - strike)

    vol_surge_at_g3 = None
    premium_score   = None
    vol_at_g3_raw   = None
    avg_vol_20d_g3  = None

    if vols is not None and len(vols) > g3_idx:
        vol_at_g3_raw  = float(vols[g3_idx])
        window         = vols[max(0, g3_idx - 20):g3_idx]
        avg_vol_20d_g3 = float(window.mean()) if len(window) > 0 else None
        if avg_vol_20d_g3 and avg_vol_20d_g3 > 0:
            vol_surge_at_g3 = round(vol_at_g3_raw / avg_vol_20d_g3, 2)
            premium_score   = round(vol_surge_at_g3 * entry_ce_premium, 2)

    threshold    = float(cfg.get("premium_vol_surge_threshold", 1.5))
    is_high_prem = bool(vol_surge_at_g3 is not None and vol_surge_at_g3 >= threshold)

    return {
        "entry_date":        str(entry_date),
        "expiry_date":       str(expiry_date),
        "actual_data_date":  str(actual_expiry),
        "entry_price":       round(entry_price, 2),
        "surge_high":        round(surge_high, 2),
        "strike":            int(strike),
        "otm_idx":           primary_v["otm_idx"],
        "otm_label":         f"+{primary_v['otm_idx']} OTM",
        "entry_ce_premium":  entry_ce_premium,
        "expiry_close":      round(expiry_close, 2),
        "intrinsic_value":   round(intrinsic_at_expiry, 2),
        "iv_sigma":          round(sigma * 100, 1),
        "T_days":            (expiry_date - entry_date).days,
        "expiry_mode":       expiry_mode,
        "expiry_label":      expiry_mode,
        "result":            result,
        "pnl_per_unit":      round(pnl, 2),
        "return_pct":        round(pnl / entry_ce_premium * 100, 1) if entry_ce_premium > 0 else 0,
        "bcs_hedge_strike":  primary_v["bcs_hedge_strike"],
        "bcs_hedge_prem":    primary_v["bcs_hedge_prem"],
        "bcs_net_premium":   primary_v["bcs_net_premium"],
        "bcs_max_profit":    primary_v["bcs_max_profit"],
        "bcs_max_loss":      primary_v["bcs_max_loss"],
        "bcs_breakeven":     primary_v["bcs_breakeven"],
        "bcs_risk_reward":   primary_v["bcs_risk_reward"],
        "vol_at_g3":         int(vol_at_g3_raw) if vol_at_g3_raw is not None else None,
        "avg_vol_20d_g3":    int(avg_vol_20d_g3) if avg_vol_20d_g3 is not None else None,
        "vol_surge_at_g3":   vol_surge_at_g3,
        "premium_score":     premium_score,
        "is_high_prem":      is_high_prem,
        "strike_variants":   strike_variants,
    }


# ─────────────────────────────────────────────────────────────────────────────
def _expiry_side(
    closes: np.ndarray, dates: pd.DatetimeIndex,
    g3_idx: int, surge_high: float, base_cfg: Dict,
    forced_mode: str,
    vols: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """
    Evaluate the CE sell trade for a single specific expiry month.
    Returns a compact dict with only the expiry-specific fields.
    Returns None if there is no historical data at/after that expiry.
    """
    cfg_override = {**base_cfg, "expiry_mode": forced_mode}
    t = evaluate_trade(closes, dates, g3_idx, surge_high, cfg_override, vols=vols)
    if t is None:
        return None
    return {
        "expiry_date":       t["expiry_date"],
        "actual_data_date":  t["actual_data_date"],
        "T_days":            t["T_days"],
        "entry_ce_premium":  t["entry_ce_premium"],
        "expiry_close":      t["expiry_close"],
        "intrinsic_value":   t["intrinsic_value"],
        "result":            t["result"],
        "pnl_per_unit":      t["pnl_per_unit"],
        "return_pct":        t["return_pct"],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  WALK-FORWARD ENGINE
# ─────────────────────────────────────────────────────────────────────────────
def backtest_ticker(
    ticker: str, cfg: Dict, years: int = 2, verbose: bool = False
) -> Dict:
    """
    Full walk-forward backtest for a single ticker.
    Returns summary + list of individual trades.

    OPTIMISED: O(N × recency) instead of the original O(N³).

    Old approach:
      for every today_idx (≈440):            ← O(N)
        detect_breakdown():
          for wsize (up to 440):             ← O(N)
            for start (up to 440):           ← O(N)
              check_surge_continuity()       ← O(wsize) each
    → ~850 million Python iterations for RELIANCE 2-year run

    New approach:
      Step 1 — 3 linear precompute passes    ← O(N)
      Step 2 — one-pass breakdown scan,
               surge search only for the
               ~10-20 actual breakdowns      ← O(N + breaks × recency²)
      Step 3 — walk-forward = dict lookups   ← O(N)
    → ~50,000 iterations; RELIANCE < 5 s
    """
    df = fetch_history(ticker, years)
    if df is None or len(df) < MIN_WARMUP + 10:
        return {"ticker": ticker, "error": "Insufficient data", "trades": []}

    closes = df["Close"].values.astype(float)
    highs  = df["High"].values.astype(float)
    lows   = df["Low"].values.astype(float)
    opens  = df["Open"].values.astype(float)
    vols   = df["Volume"].values.astype(float)
    dates  = df.index
    n      = len(closes)

    print(f"[{ticker}] {n} candles from {dates[0].date()} to {dates[-1].date()}")

    # ── Config ───────────────────────────────────────────────────────────────
    min_drop      = float(cfg.get("min_drop_percent", 0.1))
    min_vol_ratio = float(cfg.get("min_breakdown_volume_ratio", 0.5))
    recency       = int(cfg.get("surge_recency_days", 45))
    min_gain      = float(cfg.get("min_gain_percent", 18.0))
    min_green_cfg = int(cfg.get("min_green_candles", 2))
    ema_period    = int(cfg.get("ema_period", 9))

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 1 — Precompute helper arrays  (O(N) total, done ONCE)
    # ═════════════════════════════════════════════════════════════════════════

    # 1a. Continuity DP
    #     earliest_start[i] = farthest back we can trace a continuous surge
    #     ending at candle i, where "continuous" means no candle in the
    #     window closes below the previous candle's low.
    #     This single array replaces the inner check_surge_continuity loop.
    earliest_start = np.empty(n, dtype=np.intp)
    earliest_start[0] = 0
    for i in range(1, n):
        earliest_start[i] = earliest_start[i - 1] if closes[i] >= lows[i - 1] else i

    # 1b. Prefix sum of green candles
    #     green_count(start, end) = green_prefix[end+1] - green_prefix[start]
    #     Turns the green-candle count from O(window) to O(1).
    green_prefix = np.zeros(n + 1, dtype=np.int32)
    green_prefix[1:] = np.cumsum(closes > opens)

    # 1c. Rolling 20-day average volume via cumsum  → O(N), no per-candle loop
    vol_cs = np.empty(n + 1)
    vol_cs[0] = 0.0
    np.cumsum(vols, out=vol_cs[1:])
    avg_vol_20 = np.empty(n)
    for i in range(n):
        j = max(0, i - 20)
        cnt = i - j
        avg_vol_20[i] = (vol_cs[i] - vol_cs[j]) / cnt if cnt > 0 else 0.0

    # 1d. Full-series EMA — computed ONCE instead of once-per-breakdown
    ema_full = compute_ema(closes, ema_period)

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 2 — One-pass breakdown detection + surge window search
    #
    # Old code called detect_breakdown() for every today_idx (440 calls).
    # Each call ran an O(n²) double loop + O(n) check_surge_continuity →
    # the SAME breakdown candle triggered the expensive surge search 45×
    # (once per today_idx within lookback window).
    #
    # New: scan candles once; the O(recency²) surge search only fires for
    # the ~10-20 genuine breakdown candles found in a 2-year history.
    # ═════════════════════════════════════════════════════════════════════════
    precomputed_signals: Dict[int, Dict] = {}
    seen_bd_dates: set = set()

    for bd_idx in range(MIN_WARMUP, n):
        # ── Breakdown candle check (O(1) per candle) ──────────────────────
        if closes[bd_idx] >= lows[bd_idx - 1]:
            continue
        drop = (lows[bd_idx - 1] - closes[bd_idx]) / lows[bd_idx - 1] * 100
        if drop < min_drop:
            continue
        if avg_vol_20[bd_idx] <= 0:
            continue
        v_ratio = vols[bd_idx] / avg_vol_20[bd_idx]
        if v_ratio < min_vol_ratio:
            continue

        bd_date_str = str(dates[bd_idx].date())
        if bd_date_str in seen_bd_dates:
            continue
        seen_bd_dates.add(bd_date_str)

        # ── Surge window search — O(recency²) but only ~10-20 times total ─
        #    For each possible surge end (within recency before breakdown),
        #    use earliest_start[] to skip non-continuous windows entirely,
        #    and green_prefix[] for O(1) green-candle counts.
        search_start = max(0, bd_idx - recency)
        best_gain    = 0.0
        best_window  = None
        best_greens  = 0

        for end_idx in range(bd_idx - 1, search_start - 1, -1):
            cont_start  = int(earliest_start[end_idx])
            valid_start = max(cont_start, search_start)
            if valid_start >= end_idx:
                continue   # no room for a multi-candle window

            for start_idx in range(valid_start, end_idx):
                gain = (closes[end_idx] - closes[start_idx]) / closes[start_idx] * 100
                if gain < min_gain:
                    continue
                gc = int(green_prefix[end_idx + 1] - green_prefix[start_idx])
                if gc < min_green_cfg:
                    continue
                if gain > best_gain:
                    best_gain   = gain
                    best_window = (start_idx, end_idx)
                    best_greens = gc

        if best_window is None:
            continue

        surge_start_idx, surge_end_idx = best_window
        surge_high = float(np.max(highs[surge_start_idx:surge_end_idx + 1]))

        if verbose:
            print(f"  Gate 1 @ {bd_date_str}  surge_high={surge_high:.2f}  "
                  f"drop={drop:.2f}%  vol={v_ratio:.2f}x")

        precomputed_signals[bd_idx] = {
            "breakdown_idx":    bd_idx,
            "surge_high":       round(surge_high, 2),
            "yesterday_high":   round(float(highs[bd_idx - 1]), 2),
            "yesterday_low":    round(float(lows[bd_idx - 1]), 2),
            "drop_pct":         round(drop, 2),
            "volume_ratio":     round(v_ratio, 2),
            "surge_gain_pct":   round(best_gain, 2),
            "surge_candles":    best_greens,
            "surge_start_idx":  surge_start_idx,
            "surge_end_idx":    surge_end_idx,
            "ema_at_breakdown": round(float(ema_full[bd_idx]), 2),
        }

    # ═════════════════════════════════════════════════════════════════════════
    # STEP 3 — Walk-forward trade simulation (O(N) — just dict lookups)
    #          Gate 2 uses precomputed ema_full/avg_vol_20 (no recompute).
    #          Gate 3 and trade evaluation are unchanged.
    # ═════════════════════════════════════════════════════════════════════════
    trades:          List[Dict] = []
    seen_trade_keys: set        = set()

    rec_otm_idx = int(cfg.get("_recommended_otm_idx", 1))
    for bd_idx, bd_info in sorted(precomputed_signals.items()):
        surge_high  = bd_info["surge_high"]
        bd_date_str = str(dates[bd_idx].date())

        # ── Gate 2: inline with precomputed EMA — avoids recomputing full EMA
        # Start at bd_idx+1: the breakdown candle itself must NOT satisfy Gate 2.
        # Gate 2 must be a *subsequent* candle confirming EMA rejection after the
        # breakdown.  Starting at bd_idx was the bug that caused Gate1==Gate2 dates.
        g2_idx    = None
        g2_avgvol = float(np.mean(vols[max(0, bd_idx - 20):bd_idx])) if bd_idx > 0 else 0.0
        if g2_avgvol > 0:
            for i in range(bd_idx + 1, n):   # ← +1 fix: skip the breakdown candle itself
                if closes[i] < ema_full[i] and vols[i] / g2_avgvol >= min_vol_ratio:
                    g2_idx = i
                    break

        if g2_idx is None:
            if verbose:
                print(f"    Gate 2 NOT confirmed — skip")
            continue

        g2_date = str(dates[g2_idx].date())
        if verbose:
            print(f"    Gate 2 @ {g2_date}")
        # Recompute surge_high: max high from recency window BEFORE breakdown only.
        # DO NOT extend to g2_idx — Gate 2 can occur days/weeks after breakdown,
        # and including that span picks up stale post-crash highs that are not true
        # surge resistance (root cause of phantom Gate 3 on recovery rallies).
        lookback_start = max(0, bd_idx - recency)
        surge_high = float(np.max(highs[lookback_start:bd_idx + 1]))

        if verbose:
            print(f"    Surge-high (pre-EMA-break, {recency}d lookback) → {surge_high:.2f}")

        # ── Gate 3 (unchanged) ───────────────────────────────────────────────
        g3_idx = check_gate3(closes, highs, g2_idx, surge_high, cfg, dates=dates)
        if g3_idx is None:
            if verbose:
                print(f"    Gate 3 never triggered (breakout or no retest)")
            continue

        g3_date   = str(dates[g3_idx].date())
        trade_key = (round(surge_high, 2), g3_date)
        if trade_key in seen_trade_keys:
            if verbose:
                print(f"    Duplicate trade episode — skip")
            continue
        seen_trade_keys.add(trade_key)

        if verbose:
            print(f"    Gate 3 @ {g3_date}  price={closes[g3_idx]:.2f}")

        # ── Trade evaluation: pass EV-optimal OTM offset ──────────────────────
        trade = evaluate_trade(
            closes, dates, g3_idx, surge_high, cfg,
            vols=vols,
            recommended_otm_idx=rec_otm_idx,
        )
        if trade is None:
            if verbose:
                print(f"    Trade evaluation skipped (no expiry data or zero premium)")
            continue

        cur_side = _expiry_side(closes, dates, g3_idx, surge_high, cfg, "current", vols=vols)
        nxt_side = _expiry_side(closes, dates, g3_idx, surge_high, cfg, "next",    vols=vols)

        # ── Fair Value enrichment (Gate 3 trade context) ─────────────────────
        # Fetch once per ticker (cached); attach composite + per-model fields.
        # fv_valuation_bucket: UNDERVALUED = stock below fair → CE likely to
        # continue falling → historically better WIN setup.
        # OVERVALUED = stock above fair → CE may stay elevated → riskier.
        fv = _get_fair_value(ticker)

        trade.update({
            "ticker":         ticker,
            "gate1_date":     bd_date_str,
            "gate2_date":     g2_date,
            "gate3_date":     g3_date,
            "surge_gain_pct": bd_info["surge_gain_pct"],
            "surge_candles":  bd_info["surge_candles"],
            "volume_ratio":   bd_info["volume_ratio"],
            "cur_expiry":     cur_side,
            "nxt_expiry":     nxt_side,
            # ── Fair Value fields (None if sma_router unavailable or scrape failed)
            **fv,
        })
        trades.append(trade)

        if verbose:
            tag = "✅ WIN" if trade["result"] == "WIN" else "❌ LOSS"
            print(f"    {tag}  strike={trade['strike']}  "
                  f"entry_ce={trade['entry_ce_premium']}  "
                  f"expiry_close={trade['expiry_close']}  "
                  f"P&L={trade['pnl_per_unit']}")

    # ── Summary (identical structure to original) ─────────────────────────────
    total   = len(trades)
    wins    = sum(1 for t in trades if t["result"] == "WIN")
    losses  = total - wins
    avg_pnl     = round(sum(t["pnl_per_unit"] for t in trades) / total, 2) if total > 0 else 0.0
    avg_premium = round(sum(t["entry_ce_premium"] for t in trades) / total, 2) if total > 0 else 0.0

    threshold     = float(cfg.get("premium_vol_surge_threshold", 1.5))
    hi_vol_trades = [t for t in trades if (t.get("vol_surge_at_g3") or 0) >= threshold]
    lo_vol_trades = [t for t in trades if (t.get("vol_surge_at_g3") or 0) <  threshold]

    def _cohort_stats(cohort: list) -> dict:
        nc = len(cohort)
        if nc == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "accuracy_pct": None,
                    "avg_premium": None, "avg_pnl": None, "avg_prem_score": None}
        w      = sum(1 for t in cohort if t["result"] == "WIN")
        prem   = round(sum(t["entry_ce_premium"] for t in cohort) / nc, 2)
        pnl    = round(sum(t["pnl_per_unit"]     for t in cohort) / nc, 2)
        scores = [t["premium_score"] for t in cohort if t.get("premium_score") is not None]
        return {
            "trades":         nc,
            "wins":           w,
            "losses":         nc - w,
            "accuracy_pct":   round(w / nc * 100, 1),
            "avg_premium":    prem,
            "avg_pnl":        pnl,
            "avg_prem_score": round(sum(scores) / len(scores), 2) if scores else None,
        }

    def _side_summary(key: str) -> Dict:
        sides = [t[key] for t in trades if t.get(key)]
        ns    = len(sides)
        if ns == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "accuracy_pct": None,
                    "avg_premium": None, "avg_pnl": None, "total_pnl": None}
        w   = sum(1 for s in sides if s["result"] == "WIN")
        pnl = sum(s["pnl_per_unit"] for s in sides)
        return {
            "trades":       ns,
            "wins":         w,
            "losses":       ns - w,
            "accuracy_pct": round(w / ns * 100, 1),
            "avg_premium":  round(sum(s["entry_ce_premium"] for s in sides) / ns, 2),
            "avg_pnl":      round(pnl / ns, 2),
            "total_pnl":    round(pnl, 2),
        }

    # ── Max OTM Gain Pact: per-offset bucket stats ────────────────────────────
    # Group all strike_variants across every trade by otm_idx (1, 2, 3 …).
    # For each offset bucket compute: win_rate, avg_premium, avg_pnl, EV.
    # EV = avg_premium × (wins / total)  — the expected rupees collected per trade.
    # The bucket with the highest EV is the "recommended" strike offset for
    # this ticker, which the live screener can read from the cache.
    from collections import defaultdict as _dd
    _buckets: Dict[int, list] = _dd(list)
    for t in trades:
        for v in t.get("strike_variants", []):
            _buckets[v["otm_idx"]].append(v)

    otm_bucket_stats: Dict[str, Dict] = {}
    best_ev        = -1.0
    best_otm_idx   = 1          # default to +1 OTM if no variants exist
    for idx in sorted(_buckets.keys()):
        variants = _buckets[idx]
        n_v  = len(variants)
        w_v  = sum(1 for v in variants if v["result"] == "WIN")
        prem_v = round(sum(v["premium"] for v in variants) / n_v, 2)
        pnl_v  = round(sum(v["pnl"]     for v in variants) / n_v, 2)
        wr_v   = round(w_v / n_v * 100, 1)
        ev_v   = round(prem_v * (w_v / n_v), 2)
        otm_bucket_stats[str(idx)] = {
            "otm_idx":     idx,
            "label":       f"+{idx} OTM",
            "trades":      n_v,
            "wins":        w_v,
            "losses":      n_v - w_v,
            "win_rate":    wr_v,
            "avg_premium": prem_v,
            "avg_pnl":     pnl_v,
            "ev":          ev_v,
        }
        if ev_v > best_ev:
            best_ev      = ev_v
            best_otm_idx = idx

    recommended_otm = {
        "otm_idx":  best_otm_idx,
        "label":    f"+{best_otm_idx} OTM",
        "ev":       round(best_ev, 2),
        "win_rate": otm_bucket_stats[str(best_otm_idx)]["win_rate"] if otm_bucket_stats else None,
        "avg_premium": otm_bucket_stats[str(best_otm_idx)]["avg_premium"] if otm_bucket_stats else None,
    } if otm_bucket_stats else None

    # ── Fair Value cohort summary ─────────────────────────────────────────────
    def _fv_cohort(bucket: str) -> dict:
        cohort = [t for t in trades if t.get("fv_valuation_bucket") == bucket]
        nc = len(cohort)
        if nc == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "accuracy_pct": None,
                    "avg_pnl": None, "avg_premium": None}
        w   = sum(1 for t in cohort if t["result"] == "WIN")
        pnl = round(sum(t["pnl_per_unit"] for t in cohort) / nc, 2)
        prem = round(sum(t["entry_ce_premium"] for t in cohort) / nc, 2)
        return {
            "trades":       nc,
            "wins":         w,
            "losses":       nc - w,
            "accuracy_pct": round(w / nc * 100, 1),
            "avg_pnl":      pnl,
            "avg_premium":  prem,
        }

    # Representative FV snapshot for the ticker (from first successful trade)
    fv_ticker_snapshot = None
    for t in trades:
        if t.get("fv_composite_fair") is not None:
            fv_ticker_snapshot = {
                "composite_fair_price":  t["fv_composite_fair"],
                "composite_gain_pct":    t["fv_composite_gain_pct"],
                "valuation_bucket":      t["fv_valuation_bucket"],
                "model_count":           t.get("fv_model_count"),
                "op_fair":               t.get("fv_op_fair"),
                "sales_fair":            t.get("fv_sales_fair"),
                "ttm_fair":              t.get("fv_ttm_fair"),
            }
            break

    return {
        "ticker": ticker,
        "summary": {
            "total_trades":       total,
            "wins":               wins,
            "losses":             losses,
            "accuracy_pct":       round(wins / total * 100, 1) if total > 0 else 0.0,
            "avg_pnl_per_unit":   avg_pnl,
            "avg_premium":        avg_premium,
            "total_pnl":          round(sum(t["pnl_per_unit"] for t in trades), 2),
            "premium_analysis": {
                "threshold":       threshold,
                "high_vol_cohort": _cohort_stats(hi_vol_trades),
                "normal_cohort":   _cohort_stats(lo_vol_trades),
            },
            "dual_expiry_comparison": {
                "current_month": _side_summary("cur_expiry"),
                "next_month":    _side_summary("nxt_expiry"),
            },
            # Max OTM Gain Pact
            "otm_bucket_stats":   otm_bucket_stats,
            "recommended_otm":    recommended_otm,
            # ── Fair Value analysis
            "fair_value_snapshot": fv_ticker_snapshot,
            "fair_value_cohorts": {
                "undervalued": _fv_cohort("UNDERVALUED"),
                "fair":        _fv_cohort("FAIR"),
                "overvalued":  _fv_cohort("OVERVALUED"),
            },
        },
        "trades": trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  MAX OTM GAIN PACT — MODULE-LEVEL AGGREGATE HELPERS
#  These must be at module level so backtest_multiple() can call them during
#  its two-pass flow (before the nested helpers inside the function are defined).
# ─────────────────────────────────────────────────────────────────────────────
def _agg_otm_buckets(ticker_results: list) -> Dict:
    """Aggregate strike_variants across all tickers into per-offset bucket stats."""
    from collections import defaultdict as _dd2
    _bkt: Dict[int, list] = _dd2(list)
    for r in ticker_results:
        for t in r.get("trades", []):
            for v in t.get("strike_variants", []):
                _bkt[v["otm_idx"]].append(v)
    out: Dict[str, Dict] = {}
    for idx in sorted(_bkt.keys()):
        vv  = _bkt[idx]
        n_v = len(vv)
        w_v = sum(1 for v in vv if v["result"] == "WIN")
        prem_v = round(sum(v["premium"] for v in vv) / n_v, 2)
        pnl_v  = round(sum(v["pnl"]     for v in vv) / n_v, 2)
        wr_v   = round(w_v / n_v * 100, 1)
        ev_v   = round(prem_v * (w_v / n_v), 2)
        out[str(idx)] = {
            "otm_idx":     idx,
            "label":       f"+{idx} OTM",
            "trades":      n_v,
            "wins":        w_v,
            "losses":      n_v - w_v,
            "win_rate":    wr_v,
            "avg_premium": prem_v,
            "avg_pnl":     pnl_v,
            "ev":          ev_v,
        }
    return out


def _agg_best_otm(ticker_results: list) -> Optional[Dict]:
    """Return the OTM offset with the highest EV across all tickers."""
    buckets = _agg_otm_buckets(ticker_results)
    if not buckets:
        return None
    best = max(buckets.values(), key=lambda x: x["ev"])
    return {
        "otm_idx":     best["otm_idx"],
        "label":       best["label"],
        "ev":          best["ev"],
        "win_rate":    best["win_rate"],
        "avg_premium": best["avg_premium"],
    }


def backtest_multiple(
    tickers: List[str], cfg: Dict, years: int = 2, verbose: bool = False
) -> Dict:
    """
    Runs the walk-forward backtest for all tickers with Max OTM Gain Pact.

    otm_selection_mode controls the primary strike:
      "safest"  → deepest OTM with tradable premium (single pass — deterministic)
      "ev"      → two-pass: Pass 1 builds EV table, Pass 2 re-runs with EV-best offset
      "default" → always +1 OTM (single pass)
    """
    sel_mode = str(cfg.get("otm_selection_mode", "safest")).lower()

    if sel_mode == "ev":
        # ── Pass 1: build EV table with +1 OTM baseline ──────────────────────
        cfg_pass1 = {**cfg, "_recommended_otm_idx": 1}
        results = []
        for tkr in tickers:
            print(f"\n{'='*60}\nBacktesting {tkr} [Pass 1 — EV table build]...\n{'='*60}")
            results.append(backtest_ticker(tkr, cfg_pass1, years=years, verbose=verbose))

        pass1_best   = _agg_best_otm(results)
        best_otm_idx = pass1_best["otm_idx"] if pass1_best else 1

        # ── Pass 2: re-run with EV-optimal strike ────────────────────────────
        if best_otm_idx > 1:
            print(f"\n{'='*60}\nPass 2 — EV-optimal +{best_otm_idx} OTM\n{'='*60}")
            cfg_pass2 = {**cfg, "_recommended_otm_idx": best_otm_idx}
            results = []
            for tkr in tickers:
                print(f"  Re-backtesting {tkr}...")
                results.append(backtest_ticker(tkr, cfg_pass2, years=years, verbose=verbose))
        else:
            print(f"\n[OTM Pact] EV-optimal is already +1 OTM — no re-run needed")

    else:
        # ── Single pass: "safest" or "default" — selection is deterministic ──
        mode_label = "SAFEST (deepest OTM)" if sel_mode == "safest" else "DEFAULT (+1 OTM)"
        results = []
        for tkr in tickers:
            print(f"\n{'='*60}\nBacktesting {tkr} [{mode_label}]...\n{'='*60}")
            results.append(backtest_ticker(tkr, cfg, years=years, verbose=verbose))

    # Aggregate — overall
    all_trades  = [t for r in results for t in r.get("trades", [])]
    total       = len(all_trades)
    wins        = sum(1 for t in all_trades if t["result"] == "WIN")
    losses      = total - wins
    accuracy    = round(wins / total * 100, 1) if total > 0 else 0.0
    total_pnl   = round(sum(t["pnl_per_unit"] for t in all_trades), 2)

    # Aggregate — Premium Surge cohort (cross-ticker)
    threshold    = float(cfg.get("premium_vol_surge_threshold", 1.5))
    hi_all  = [t for t in all_trades if (t.get("vol_surge_at_g3") or 0) >= threshold]
    lo_all  = [t for t in all_trades if (t.get("vol_surge_at_g3") or 0) <  threshold]

    def _agg_cohort(cohort: list) -> dict:
        n = len(cohort)
        if n == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "accuracy_pct": None,
                    "avg_premium": None, "avg_pnl": None, "avg_prem_score": None}
        w     = sum(1 for t in cohort if t["result"] == "WIN")
        prem  = round(sum(t["entry_ce_premium"] for t in cohort) / n, 2)
        pnl   = round(sum(t["pnl_per_unit"]     for t in cohort) / n, 2)
        scores = [t["premium_score"] for t in cohort if t.get("premium_score") is not None]
        return {
            "trades":         n,
            "wins":           w,
            "losses":         n - w,
            "accuracy_pct":   round(w / n * 100, 1),
            "avg_premium":    prem,
            "avg_pnl":        pnl,
            "avg_prem_score": round(sum(scores) / len(scores), 2) if scores else None,
        }

    # Aggregate dual-expiry across all tickers
    def _agg_side(key: str) -> Dict:
        sides = [t[key] for r in results for t in r.get("trades", []) if t.get(key)]
        n = len(sides)
        if n == 0:
            return {"trades": 0, "wins": 0, "losses": 0, "accuracy_pct": None,
                    "avg_premium": None, "avg_pnl": None, "total_pnl": None}
        w   = sum(1 for s in sides if s["result"] == "WIN")
        pnl = sum(s["pnl_per_unit"] for s in sides)
        return {
            "trades":       n,
            "wins":         w,
            "losses":       n - w,
            "accuracy_pct": round(w / n * 100, 1),
            "avg_premium":  round(sum(s["entry_ce_premium"] for s in sides) / n, 2),
            "avg_pnl":      round(pnl / n, 2),
            "total_pnl":    round(pnl, 2),
        }

    # (OTM aggregate helpers are module-level: _agg_otm_buckets, _agg_best_otm)

    return {
        "config":      cfg,
        "years_tested":years,
        "aggregate": {
            "tickers":       tickers,
            "total_trades":  total,
            "wins":          wins,
            "losses":        losses,
            "accuracy_pct":  accuracy,
            "total_pnl":     total_pnl,
            "avg_pnl":       round(total_pnl / total, 2) if total > 0 else 0.0,
            # Premium Surge aggregate
            "premium_analysis": {
                "threshold":       threshold,
                "high_vol_cohort": _agg_cohort(hi_all),
                "normal_cohort":   _agg_cohort(lo_all),
            },
            # Dual-expiry aggregate comparison
            "dual_expiry_comparison": {
                "current_month": _agg_side("cur_expiry"),
                "next_month":    _agg_side("nxt_expiry"),
            },
            # Max OTM Gain Pact — aggregate per-offset bucket stats (cross-ticker)
            "otm_bucket_stats":    _agg_otm_buckets(results),
            "recommended_otm":     _agg_best_otm(results),
        },
        "per_ticker": results,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  FASTAPI ROUTER  (optional — only mounted if FastAPI is available)
# ─────────────────────────────────────────────────────────────────────────────
if _has_fastapi:
    from pydantic import BaseModel as _BM

    class BacktestRequest(_BM):
        tickers:  List[str]
        years:    int        = 2
        config:   Dict       = {}
        force:    bool       = False   # True → bypass cache, always run fresh

    router = APIRouter(prefix="/api/backtest", tags=["backtest"])

    @router.post("/run")
    async def run_backtest(req: BacktestRequest):
        cfg = {**DEFAULT_CONFIG, **req.config}
        key = _cache_key(req.tickers, req.years, cfg)

        # ── Cache hit ────────────────────────────────────────────────────────
        if not req.force:
            cached = _cache_load(key)
            if cached is not None:
                cached["from_cache"] = True
                print(f"[cache] HIT  {key[:10]}…  tickers={req.tickers}")
                return JSONResponse(content=cached)

        # ── Cache miss → run full backtest ───────────────────────────────────
        print(f"[cache] MISS {key[:10]}…  tickers={req.tickers}  force={req.force}")
        try:
            result = backtest_multiple(req.tickers, cfg, years=req.years, verbose=False)
            result["from_cache"] = False
            result["cached_at"]  = datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
            _cache_save(key, result)
            return JSONResponse(content=result)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @router.get("/config")
    async def backtest_config():
        return DEFAULT_CONFIG

    @router.get("/cache")
    async def list_cache():
        """List all cached backtest results with metadata."""
        return JSONResponse(content={"entries": _cache_list()})

    @router.delete("/cache/{key}")
    async def delete_cache_entry(key: str):
        """Delete a single cache entry by its MD5 key."""
        deleted = _cache_delete(key)
        return JSONResponse(content={"deleted": deleted, "key": key})

    @router.delete("/cache")
    async def clear_all_cache():
        """Wipe the entire cache directory."""
        count = 0
        if CACHE_DIR.exists():
            for p in CACHE_DIR.glob("*.json"):
                p.unlink()
                count += 1
        return JSONResponse(content={"cleared": count})

    @router.get("/candles")
    async def get_candles(ticker: str, from_date: str, to_date: str):
        """
        Return OHLCV candles for ticker between from_date and to_date (inclusive).
        Used by the frontend to render per-trade candlestick charts.
        """
        try:
            from_d = date.fromisoformat(from_date)
            to_d   = date.fromisoformat(to_date)
            years_needed = max(1, math.ceil((to_d - from_d).days / 300) + 1)
            df = fetch_history(ticker, years=years_needed + 1)
            if df is None or df.empty:
                return JSONResponse(status_code=404, content={"error": f"No data for {ticker}"})
            candles = []
            for dt, row in df.iterrows():
                d = dt.date()
                if d < from_d or d > to_d:
                    continue
                candles.append({
                    "date":   str(d),
                    "open":   round(float(row["Open"]),   2),
                    "high":   round(float(row["High"]),   2),
                    "low":    round(float(row["Low"]),    2),
                    "close":  round(float(row["Close"]),  2),
                    "volume": int(row["Volume"]),
                })
            return JSONResponse(content={"ticker": ticker, "candles": candles})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})


# ─────────────────────────────────────────────────────────────────────────────
#  STANDALONE CLI
# ─────────────────────────────────────────────────────────────────────────────
def _print_results(result: Dict):
    agg = result.get("aggregate", {})
    print(f"\n{'═'*70}")
    print(f"  BACKTEST RESULTS  ({result.get('years_tested', '?')} years)")
    print(f"{'═'*70}")
    print(f"  Tickers  : {', '.join(agg.get('tickers', []))}")
    print(f"  Trades   : {agg.get('total_trades', 0)}")
    print(f"  Wins     : {agg.get('wins', 0)}")
    print(f"  Losses   : {agg.get('losses', 0)}")
    print(f"  Accuracy : {agg.get('accuracy_pct', 0):.1f}%")
    print(f"  Avg P&L  : ₹{agg.get('avg_pnl', 0):.2f} per unit")
    print(f"  Total P&L: ₹{agg.get('total_pnl', 0):.2f} (sum across all trades)")
    print(f"{'═'*70}\n")

    for tkr_res in result.get("per_ticker", []):
        tkr = tkr_res["ticker"]
        s   = tkr_res.get("summary", {})
        err = tkr_res.get("error")
        if err:
            print(f"  {tkr}: ERROR — {err}")
            continue
        print(f"  {tkr}: {s.get('total_trades',0)} trades  "
              f"Acc={s.get('accuracy_pct',0)}%  "
              f"AvgP&L=₹{s.get('avg_pnl_per_unit',0):.2f}  "
              f"AvgPremium=₹{s.get('avg_premium',0):.2f}")

        for t in tkr_res.get("trades", []):
            tag = "✅" if t["result"] == "WIN" else "❌"
            print(
                f"    {tag} G1={t['gate1_date']} G3={t['gate3_date']}"
                f"  Surge={t['surge_high']}  Strike={t['strike']}"
                f"  Premium=₹{t['entry_ce_premium']}"
                f"  ExpiryClose={t['expiry_close']}"
                f"  P&L=₹{t['pnl_per_unit']}"
                f"  ({t['result']})"
            )
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NSE Momentum Loss Screener — Backtesting Engine"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=["RELIANCE"],
        help="NSE ticker symbols (without .NS suffix)"
    )
    parser.add_argument(
        "--years", type=int, default=2,
        help="Years of historical data to test over (default: 2)"
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config.json (uses defaults if omitted)"
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Save results as JSON to this path"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-signal debug output"
    )
    args = parser.parse_args()

    cfg = DEFAULT_CONFIG.copy()
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            cfg.update(json.load(f))

    result = backtest_multiple(args.tickers, cfg, years=args.years, verbose=args.verbose)
    _print_results(result)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved → {args.out}")