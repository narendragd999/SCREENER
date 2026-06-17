"""
NSE Momentum Loss Screener — FastAPI Backend
============================================
Run: uvicorn main:app --host 0.0.0.0 --port 8000 --reload

THESIS
------
1. Stock surges >= N% continuously (green candles with higher highs;
   red candles allowed only if they do NOT close below the previous
   candle's low — i.e. the higher-low structure stays intact).
2. Surge must have ended within `surge_recency_days` trading sessions
   of today — ensures the breakdown happens right after the surge,
   while market memory of those highs is still fresh.
3. TODAY's candle closes BELOW the previous day's LOW with volume
   >= `min_breakdown_volume_ratio` x 20-day average  -> momentum loss.
4. Today's close must be >= `min_drop_percent` below previous low
   (filters trivial tick-below-low noise like -0.18%).
5. The EMA breach is a ONE-TIME trend-weakness gate — but it does NOT
   have to happen on the same candle as the breakdown. The screener
   looks back up to `surge_recency_days` sessions to find a valid
   breakdown candle (close below prev low + volume), then checks if
   the 9-EMA was breached on THAT candle or any candle after it up
   to today. Once the signal is saved, the EMA gate is skipped on all
   future re-scans — trend damage is already confirmed.
6. The SURGE HIGH (peak of the entire surge window) is the true
   resistance ceiling — this is where original momentum exhausted.
   Even after breakdown + partial recovery, this level rejects price
   again. The CE strike is anchored above this level.
7. When price rallies back within `price_proximity_percent` of the
   surge high -> CE premium is elevated -> SELL CE above that
   resistance. Theta decay + failed retest = profit.
"""

import os, json, time, random, asyncio, threading, uuid, io
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import pytz
import requests
import yfinance as yf
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger

from fastapi import FastAPI, BackgroundTasks, UploadFile, File, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Sub-routers (Option Charts + SMA Screener)
from option_charts_router import router as oc_router
from sma_router import router as sma_router

# ADD THIS:
from backtest import router as bt_router

BASE_DIR = Path(__file__).parent

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────
TICKERS_FILE   = "tickers.csv"
CONFIG_FILE    = "config.json"
SIGNALS_FILE   = "signals.json"
PROXIMITY_FILE = "proximity_alerts.json"
SCAN_LOG_FILE  = "scan_log.json"
TRACKER_FILE        = "strike_tracker.json"
PREMIUM_ZONE_FILE   = "premium_zone.json"

IST = pytz.timezone("Asia/Kolkata")

NSE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36"
    ),
    "Accept":          "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.nseindia.com/option-chain",
    "Sec-Fetch-Site":  "same-origin",
    "Sec-Fetch-Mode":  "cors",
    "Connection":      "keep-alive",
}

_nse_session:         Optional[requests.Session] = None
_nse_session_created: Optional[datetime]         = None
_nse_lock = threading.Lock()
NSE_SESSION_TTL_SECS = 25 * 60   # proactive refresh after 25 min

# ─────────────────────────────────────────────────────────────
#  JOB MANAGEMENT
# ─────────────────────────────────────────────────────────────
jobs: Dict[str, Dict] = {}
jobs_lock = threading.Lock()

def _prune_old_jobs():
    """Drop completed jobs older than 2 hours to prevent memory growth."""
    cutoff = datetime.now(IST) - timedelta(hours=2)
    with jobs_lock:
        stale = [
            jid for jid, j in jobs.items()
            if j.get("status") in ("done", "error") and j.get("created_at")
            and datetime.strptime(j["created_at"], "%Y-%m-%d %H:%M:%S")
               .replace(tzinfo=IST) < cutoff
        ]
        for jid in stale:
            del jobs[jid]

def create_job() -> str:
    _prune_old_jobs()
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "status":         "running",
            "progress":       0,
            "total":          0,
            "current_ticker": "",
            "logs":           [],
            "result":         [],
            "created_at":     now_ist_str(),
        }
    return job_id

def job_log(job_id: str, msg: str, level: str = "info"):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["logs"].append({
                "time":  datetime.now(IST).strftime("%H:%M:%S"),
                "msg":   msg,
                "level": level,
            })

def job_progress(job_id: str, current: int, total: int, ticker: str):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["progress"]       = current
            jobs[job_id]["total"]          = total
            jobs[job_id]["current_ticker"] = ticker

def job_done(job_id: str, result: list, status: str = "done"):
    with jobs_lock:
        if job_id in jobs:
            jobs[job_id]["status"] = status
            jobs[job_id]["result"] = result

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # Surge detection
    "lookback_days":               30,
    "min_gain_percent":            20.0,
    "min_green_candles":           3,
    "surge_recency_days":          5,      # surge_end must be within N trading days of today
    # Breakdown (momentum loss) candle
    "min_drop_percent":            0.5,    # must close this % below prev low (not just a tick)
    "min_breakdown_volume_ratio":  1.2,    # breakdown volume / 20d avg volume
    # Trend filter
    "ema_period":                  9,
    "ema_filter_enabled":          True,   # breakdown close must be below EMA
    # Sell zone
    "price_proximity_percent":     2.0,    # within X% below yesterday_high = sell zone
    # CE option filter
    "ce_above_historical_high":    True,
    "ce_history_days":             30,
    # Signal freshness
    "max_signal_age_days":         5,      # auto-prune signals older than N days
    # Auto scan
    "auto_scan_enabled":           True,
    "auto_scan_interval_min":      15,
    "market_hours_only":           True,
    # Previous-Pullup-High filter (Check 1)
    # Finds the highest high in the window BEFORE the current surge/breakdown period.
    # That level = the "last pullup high" = prior resistance the stock previously peaked at.
    # If current price is at/above that level, the stock has broken out to new highs
    # (bullish momentum intact) — selling CE is risky → SKIP.
    #
    # Example: HINDALCO Jan 29 2026 high ~985.  If current price >= 985 × 0.995,
    # the stock is above its prior peak → skip.
    #
    # Window: (today - prev_high_lookback_days)  to  (today - prev_high_exclude_recent_days)
    # The "exclude recent" gap ensures the current surge itself is NOT counted as the prior high.
    "prev_high_filter_enabled":      True,
    "prev_high_lookback_days":       120,    # how far back the prior-high window starts
    "prev_high_exclude_recent_days": 20,     # exclude last N days (= current surge window)
    "prev_high_buffer_pct":          0.5,    # if price >= prior_high*(1-buffer%), skip ticker
    # Bear Call Spread (Check 2)
    # After finding the short CE, we also find a buy-leg N intervals higher
    # to cap max loss.  Set to 0 to use a single short CE only.
    "bear_call_spread_width_intervals": 1,   # 1 = buy the very next strike above short CE
    # Telegram
    "telegram_bot_token":          "",
    "telegram_chat_id":            "",
    # Premium Sell Zone tab
    # Minimum live-volume / 20d-avg-volume ratio required to flag a stock
    # as having a "volume surge" in the Premium Zone scan.
    "premium_min_vol_surge":       1.5,
    # How many OTM strikes above ATM to include in the option ladder
    "premium_otm_depth":           3,
}

def load_config() -> Dict:
    cfg = DEFAULT_CONFIG.copy()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            cfg.update(json.load(f))
    return cfg

def save_config(cfg: Dict):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

# ─────────────────────────────────────────────────────────────
#  PERSISTENCE
# ─────────────────────────────────────────────────────────────
def _rj(path: str, default):
    try:
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
    except Exception:
        pass
    return default

def _wj(path: str, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

load_signals   = lambda: _rj(SIGNALS_FILE, [])
save_signals   = lambda d: _wj(SIGNALS_FILE, d)
load_proximity = lambda: _rj(PROXIMITY_FILE, [])
save_proximity = lambda d: _wj(PROXIMITY_FILE, d)
load_scan_log  = lambda: _rj(SCAN_LOG_FILE, [])
save_scan_log  = lambda d: _wj(SCAN_LOG_FILE, d)
load_premium_zone  = lambda: _rj(PREMIUM_ZONE_FILE, [])
save_premium_zone  = lambda d: _wj(PREMIUM_ZONE_FILE, d)

# ─────────────────────────────────────────────────────────────
#  TELEGRAM
# ─────────────────────────────────────────────────────────────
def send_telegram(token: str, chat_id: str, text: str):
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    for i in range(0, len(text), 4096):
        chunk = text[i:i+4096].strip()
        try:
            requests.post(url, json={"chat_id": chat_id, "text": chunk,
                                     "parse_mode": "Markdown"}, timeout=10)
        except Exception as e:
            print(f"[Telegram] {e}")
        time.sleep(0.3)

# ─────────────────────────────────────────────────────────────
#  TICKERS
# ─────────────────────────────────────────────────────────────
def load_tickers() -> List[str]:
    try:
        if os.path.exists(TICKERS_FILE):
            df = pd.read_csv(TICKERS_FILE)
            if "SYMBOL" in df.columns:
                return [str(s).strip().upper() for s in df["SYMBOL"].dropna()]
    except Exception:
        pass
    return ["HDFCBANK", "RELIANCE", "TCS", "INFY", "ICICIBANK"]

# ─────────────────────────────────────────────────────────────
#  UTILITY
# ─────────────────────────────────────────────────────────────
def is_market_open() -> bool:
    now = datetime.now(IST)
    if now.weekday() >= 5:
        return False
    o = now.replace(hour=9,  minute=15, second=0, microsecond=0)
    c = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return o <= now <= c

def now_ist_str() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")

def compute_ema(values: np.ndarray, period: int) -> np.ndarray:
    """EMA via pandas ewm — matches standard charting tools."""
    return pd.Series(values, dtype=float).ewm(span=period, adjust=False).mean().values

# ─────────────────────────────────────────────────────────────
#  NSE SESSION  (TTL-based proactive refresh)
# ─────────────────────────────────────────────────────────────
def get_nse_session() -> Optional[requests.Session]:
    global _nse_session, _nse_session_created
    with _nse_lock:
        if _nse_session is not None and _nse_session_created is not None:
            age = (datetime.now(IST) - _nse_session_created).total_seconds()
            if age > NSE_SESSION_TTL_SECS:
                print(f"[NSE] Session TTL exceeded ({age:.0f}s) — refreshing")
                _nse_session         = None
                _nse_session_created = None

        if _nse_session is not None:
            return _nse_session

        s = requests.Session()
        s.headers.update(NSE_HEADERS)
        try:
            s.get("https://www.nseindia.com/", timeout=12)
            time.sleep(random.uniform(1.5, 2.5))
            s.get("https://www.nseindia.com/option-chain", timeout=12)
            time.sleep(random.uniform(2.0, 3.0))
            _nse_session         = s
            _nse_session_created = datetime.now(IST)
            return s
        except Exception as e:
            print(f"[NSE] Session init failed: {e}")
            return None

def reset_nse_session():
    global _nse_session, _nse_session_created
    with _nse_lock:
        _nse_session         = None
        _nse_session_created = None

# Adaptive NSE throttle — tracks consecutive 429/403 responses.
# If NSE starts rate-limiting, sleeps scale up automatically.
_nse_backoff = 0
_nse_backoff_lock = threading.Lock()

def _nse_ok():
    global _nse_backoff
    with _nse_backoff_lock:
        _nse_backoff = max(0, _nse_backoff - 1)

def _nse_hit():
    global _nse_backoff
    with _nse_backoff_lock:
        _nse_backoff = min(_nse_backoff + 3, 12)

def _nse_sleep(base_lo: float, base_hi: float):
    """Sleep with exponential backoff if NSE has been returning errors."""
    with _nse_backoff_lock:
        b = _nse_backoff
    extra = b * 2.0
    time.sleep(random.uniform(base_lo + extra, base_hi + extra))

# ─────────────────────────────────────────────────────────────

def fetch_option_chain(ticker: str) -> Tuple[Optional[Dict], Optional[str]]:
    session = get_nse_session()
    if session is None:
        return None, None
    try:
        r = session.get(
            f"https://www.nseindia.com/api/option-chain-contract-info?symbol={ticker}",
            timeout=12,
        )
        if r.status_code in (429, 403):
            _nse_hit()
            reset_nse_session()
            return None, None
        if r.status_code != 200 or not r.text.strip().startswith("{"):
            return None, None
        _nse_ok()
        d = r.json()
        expiries = d.get("expiryDates", []) or d.get("records", {}).get("expiryDates", [])
        if not expiries:
            return None, None

        # Skip nearest expiry if within 3 calendar days (high gamma risk)
        today_dt      = date.today()
        chosen_expiry = expiries[0]
        try:
            nearest_dt = datetime.strptime(expiries[0], "%d-%b-%Y").date()
            if (nearest_dt - today_dt).days <= 3 and len(expiries) > 1:
                chosen_expiry = expiries[1]
                print(f"[NSE] {ticker}: nearest expiry {expiries[0]} too close "
                      f"({(nearest_dt - today_dt).days}d) -> using {chosen_expiry}")
        except Exception:
            pass

        expiry = chosen_expiry
    except Exception as e:
        print(f"[NSE] Expiry error {ticker}: {e}")
        return None, None

    _nse_sleep(0.5, 1.0)  # adaptive: scales up automatically if 429s occur

    try:
        r2 = session.get(
            f"https://www.nseindia.com/api/option-chain-v3"
            f"?type=Equity&symbol={ticker}&expiry={expiry}",
            timeout=15,
        )
        if r2.status_code in (429, 403):
            _nse_hit()
            reset_nse_session()
            return None, None
        if r2.status_code != 200 or not r2.text.strip().startswith("{"):
            return None, None
        _nse_ok()
        records = r2.json().get("filtered", {}).get("data", [])
        strike_map: Dict[float, Dict] = {}
        for rec in records:
            sp  = float(rec.get("strikePrice", 0))
            ce  = rec.get("CE", {})
            ltp = ce.get("lastPrice", 0.0)
            oi  = ce.get("openInterest", 0)
            if sp > 0:
                strike_map[sp] = {"CE_ltp": float(ltp), "CE_oi": int(oi)}
        return strike_map, expiry
    except Exception as e:
        print(f"[NSE] Option chain error {ticker}: {e}")
        return None, None

# ─────────────────────────────────────────────────────────────
#  YFINANCE HELPERS
# ─────────────────────────────────────────────────────────────
# ── Per-scan yfinance cache ──────────────────────────────────────────────────
# Within a single scan job we fetch each ticker once (120 days — enough for
# both the short lookback AND the prev-pullup-high window).  Subsequent calls
# with smaller `days` values are satisfied from the cache by slicing.
# The cache is reset at the start of every scan job (see run_screen_job).
_yf_cache: Dict[str, pd.DataFrame] = {}
_yf_cache_lock = threading.Lock()

# Maximum days we ever need in one call (prev_high_lookback_days default=120)
_YF_FETCH_DAYS = 135   # 120 + 15 buffer

def _fetch_and_cache(ticker: str) -> Optional[pd.DataFrame]:
    """Fetch 120d of history for ticker, store in cache, return full frame."""
    try:
        now_ist  = datetime.now(IST)
        end_dt   = (now_ist + timedelta(days=1)).date()
        start_dt = (now_ist - timedelta(days=_YF_FETCH_DAYS)).date()
        hist = yf.Ticker(f"{ticker}.NS").history(
            start=str(start_dt), end=str(end_dt), auto_adjust=True
        )
        if hist.empty:
            return None
        if hist.index.tz is None:
            hist.index = hist.index.tz_localize("UTC")
        hist.index = hist.index.tz_convert(IST)
        hist = hist.sort_index()
        hist = hist[hist.index.date <= now_ist.date()]
        with _yf_cache_lock:
            _yf_cache[ticker] = hist
        return hist
    except Exception as e:
        print(f"[yf] History error {ticker}: {e}")
        return None

def get_price_history(ticker: str, days: int) -> Optional[pd.DataFrame]:
    """Return price history for `days` lookback. Uses per-scan cache."""
    with _yf_cache_lock:
        cached = _yf_cache.get(ticker)
    if cached is None:
        cached = _fetch_and_cache(ticker)
    if cached is None:
        return None
    # Slice to requested window
    now_ist  = datetime.now(IST)
    start_dt = (now_ist - timedelta(days=days + 15)).date()
    sliced   = cached[cached.index.date >= start_dt]
    return sliced if not sliced.empty else None

def get_current_price(ticker: str) -> Optional[float]:
    try:
        p = yf.Ticker(f"{ticker}.NS").fast_info.last_price
        if p:
            return float(p)
    except Exception:
        pass
    # Fallback: use last close from cache (avoids extra HTTP during bulk scan)
    with _yf_cache_lock:
        cached = _yf_cache.get(ticker)
    if cached is not None and not cached.empty:
        return float(cached["Close"].iloc[-1])
    return None

def get_previous_pullup_high(
    ticker: str,
    lookback_days: int = 120,
    exclude_recent_days: int = 20,
) -> Optional[float]:
    """
    Returns the highest intraday High in the 'prior window':
        from  (today - lookback_days)
        to    (today - exclude_recent_days)

    Uses the per-scan yfinance cache — NO second HTTP call.
    """
    try:
        end_dt   = date.today() - timedelta(days=exclude_recent_days)
        start_dt = date.today() - timedelta(days=lookback_days)
        if end_dt <= start_dt:
            return None
        # Pull from cache (already fetched by get_price_history)
        with _yf_cache_lock:
            hist = _yf_cache.get(ticker)
        if hist is None:
            hist = _fetch_and_cache(ticker)
        if hist is None:
            return None
        window = hist[
            (hist.index.date >= start_dt) &
            (hist.index.date <= end_dt)
        ]
        if window.empty or len(window) < 3:
            return None
        return float(window["High"].max())
    except Exception as e:
        print(f"[yf] Previous pullup high error {ticker}: {e}")
        return None

# ─────────────────────────────────────────────────────────────
#  CE HISTORICAL HIGH
# ─────────────────────────────────────────────────────────────
def get_ce_historical_high(ticker: str, strike: float, expiry_str: str,
                            days: int = 30) -> Tuple[Optional[float], int]:
    try:
        session = get_nse_session()
        if session is None:
            print(f"[{ticker}] NSE session is None")
            return None, 0

        expiry_dt = datetime.strptime(expiry_str, "%d-%b-%Y")
        exp_nse   = expiry_dt.strftime("%d-%b-%Y")
        year      = expiry_dt.year
        end_dt    = date.today()
        start_dt  = end_dt - timedelta(days=days + 5)

        url = (
            f"https://www.nseindia.com/api/historicalOR/foCPV"
            f"?from={start_dt.strftime('%d-%m-%Y')}"
            f"&to={end_dt.strftime('%d-%m-%Y')}"
            f"&instrumentType=OPTSTK"
            f"&symbol={ticker}"
            f"&year={year}"
            f"&expiryDate={exp_nse}"
            f"&optionType=CE"
            f"&strikePrice={int(strike)}"
        )
        print(f"\n{'='*60}")
        print(f"[{ticker}] NSE CE HIST URL:\n  {url}")
        print(f"  expiry_str received : {expiry_str!r}")
        print(f"  exp_nse formatted   : {exp_nse!r}")
        print(f"  year                : {year}")
        print(f"  strike (int)        : {int(strike)}")
        print(f"  date range          : {start_dt} -> {end_dt}")

        time.sleep(random.uniform(1.0, 2.0))
        r = session.get(url, timeout=15)

        print(f"[{ticker}] HTTP STATUS : {r.status_code}")
        print(f"[{ticker}] RESPONSE (first 500 chars):\n  {r.text[:500]!r}")

        if r.status_code in (403, 429):
            print(f"[{ticker}] Blocked ({r.status_code}) -- resetting session")
            reset_nse_session()
            return None, 1
        if r.status_code != 200:
            print(f"[{ticker}] Non-200 status: {r.status_code}")
            return None, 1
        if not r.text.strip().startswith("{"):
            print(f"[{ticker}] Response is NOT JSON")
            return None, 1

        data = r.json()
        print(f"[{ticker}] TOP-LEVEL KEYS : {list(data.keys())}")

        rows = (
            data.get("data")
            or data.get("Data")
            or data.get("records", {}).get("data")
            or []
        )
        print(f"[{ticker}] ROWS COUNT : {len(rows)}")
        if rows:
            print(f"[{ticker}] FIRST ROW KEYS : {list(rows[0].keys())}")
            print(f"[{ticker}] FIRST ROW DATA : {rows[0]}")

        if not rows:
            print(f"[{ticker}] Empty rows -- API returned no historical data")
            return None, 1

        ltps = []
        for row in rows:
            val = (
                row.get("FH_CLOSING_PRICE")
                or row.get("FH_LAST_TRADED_PRICE")
                or row.get("close")
                or row.get("lastPrice")
                or row.get("FH_TRADE_HIGH_PRICE")
                or row.get("CH_CLOSING_PRICE")
                or 0
            )
            try:
                v = float(str(val).replace(",", ""))
                if v > 0:
                    ltps.append(v)
            except (ValueError, TypeError):
                continue

        print(f"[{ticker}] EXTRACTED LTPs : {ltps[:10]}{'...' if len(ltps)>10 else ''}")
        if not ltps:
            print(f"[{ticker}] No valid LTP values")
            return None, 1

        high = max(ltps)
        print(f"[{ticker}] CE {days}d HIGH = Rs{high}")
        print(f"{'='*60}\n")
        return high, 1

    except Exception as e:
        import traceback
        print(f"[{ticker}] EXCEPTION in get_ce_historical_high:")
        traceback.print_exc()
        return None, 0

# ─────────────────────────────────────────────────────────────
#  STRIKE SELECTION
#  Anchored to yesterday_high (resistance ceiling), not today_close.
#  We sell a CE that only goes ITM if price breaks above resistance.
# ─────────────────────────────────────────────────────────────
def nearest_round_strike_above(price: float, strike_map: Dict) -> Optional[float]:
    liquid_above = {
        s: data for s, data in strike_map.items()
        if s > price and data["CE_ltp"] > 0
    }
    if not liquid_above:
        above = [s for s in strike_map if s > price]
        return min(above) if above else None
    sorted_strikes = sorted(liquid_above.keys(), key=lambda s: s - price)
    top3 = sorted_strikes[:3]
    round_in_top3 = [s for s in top3 if int(s) % 50 == 0]
    if round_in_top3:
        return min(round_in_top3, key=lambda s: s - price)
    return sorted_strikes[0]


# ─────────────────────────────────────────────────────────────
#  MATH-ONLY STRIKE HELPERS
#  Used when we need a strike estimate WITHOUT an NSE option-chain
#  call (e.g. when recomputing surge_high at Gate 2 confirmation).
# ─────────────────────────────────────────────────────────────
import math as _math

def nse_strike_interval(price: float) -> float:
    """NSE standard strike interval for a given stock price."""
    if price < 250:   return 10
    if price < 500:   return 20
    if price < 1000:  return 50
    if price < 2500:  return 100
    if price < 5000:  return 200
    return 500


def nearest_otm_strike_math(surge_high: float) -> float:
    """
    Nearest strike STRICTLY ABOVE surge_high using NSE standard
    intervals — no option-chain call required.
    Mirrors backtest.py::nearest_otm_strike().
    """
    interval = nse_strike_interval(surge_high)
    return _math.ceil(surge_high / interval) * interval

# ─────────────────────────────────────────────────────────────
#  SURGE CONTINUITY CHECK
#
#  A surge window [start, end] is "continuous" when every red
#  candle (close < open) inside it closes AT OR ABOVE the
#  previous candle's low.  This preserves the higher-low
#  structure while tolerating healthy pullback candles.
#
#  A red candle that closes BELOW the previous candle's low
#  is a real structural break and disqualifies the window.
# ─────────────────────────────────────────────────────────────
def check_surge_continuity(
    closes: np.ndarray,
    opens:  np.ndarray,
    lows:   np.ndarray,
    start:  int,
    end:    int,
) -> Tuple[bool, int, int]:
    """
    Returns (is_continuous, allowed_red_count, green_count).
    is_continuous=False means a red candle broke the prior low.
    """
    green_count       = 0
    allowed_red_count = 0

    for i in range(start, end + 1):
        is_green = closes[i] > opens[i]
        if is_green:
            green_count += 1
        else:
            # Red candle -- check higher-low structure
            if i > start and closes[i] < lows[i - 1]:
                # Closed below previous candle's low = structural break
                return False, allowed_red_count, green_count
            allowed_red_count += 1

    return True, allowed_red_count, green_count


# ─────────────────────────────────────────────────────────────
#  SCREENING -- STEP 1
# ─────────────────────────────────────────────────────────────
def check_surge_and_loss(ticker: str, cfg: Dict, job_id: str = "") -> Optional[Dict]:
    hist = get_price_history(ticker, cfg["lookback_days"])
    if hist is None or len(hist) < cfg["min_green_candles"] + 5:
        return None

    hist   = hist.sort_index()
    closes = hist["Close"].values.copy()
    highs  = hist["High"].values.copy()
    lows   = hist["Low"].values.copy()
    opens  = hist["Open"].values.copy()
    vols   = hist["Volume"].values.copy()
    dates  = hist.index

    if len(closes) < 3:
        return None

    # Inject live price into today's close (and high) during market hours
    live_price = get_current_price(ticker)
    if live_price is not None:
        closes[-1] = live_price
        if live_price > highs[-1]:
            highs[-1] = live_price

    # ── CHECK 1: Previous-Pullup-High filter ─────────────────────────────────
    # Goal: skip tickers where the current price has already surpassed (or is
    # sitting right at) the LAST significant swing/pullup high — the prior
    # resistance ceiling formed BEFORE the current surge.
    #
    # Why this matters: if price > prior high, the stock is in breakout mode.
    # There is no overhead resistance to reject price → CE selling is risky.
    #
    # The filter looks at a window BEFORE the recent surge period so the
    # current move is never mis-counted as the "previous high."
    #
    # Window layout (example with defaults):
    #   ├── today-120d ──────── prior-high window ──────── today-20d ──┤ today
    #                           ↑ max(High) here = prev_pullup_high
    #   └─── current surge + breakdown ─── (last ~20d, excluded) ─────┘
    lth_enabled: bool           = cfg.get("prev_high_filter_enabled", True)
    lifetime_high: Optional[float] = None   # populated below; reused in candidate dict
    if lth_enabled:
        lth_lookback = int(cfg.get("prev_high_lookback_days",       120))
        lth_exclude  = int(cfg.get("prev_high_exclude_recent_days",  20))
        lth_buffer   = float(cfg.get("prev_high_buffer_pct",          0.5))
        cur_price_for_lth = float(live_price if live_price is not None else closes[-1])

        lifetime_high = get_previous_pullup_high(ticker, lth_lookback, lth_exclude)
        if lifetime_high is not None and lifetime_high > 0:
            lth_threshold = lifetime_high * (1.0 - lth_buffer / 100.0)
            if cur_price_for_lth >= lth_threshold:
                if job_id:
                    job_log(
                        job_id,
                        f"{ticker} SKIP (Prev-Pullup-High): "
                        f"price Rs{cur_price_for_lth:.2f} >= threshold Rs{lth_threshold:.2f} "
                        f"(prev high Rs{lifetime_high:.2f} from "
                        f"{lth_exclude}-{lth_lookback}d window, buffer {lth_buffer}%)",
                        "fail",
                    )
                return None
            if job_id:
                dist_from_prev = (lifetime_high - cur_price_for_lth) / lifetime_high * 100
                job_log(
                    job_id,
                    f"{ticker} Prev-Pullup-High OK: "
                    f"price Rs{cur_price_for_lth:.2f} is {dist_from_prev:.1f}% "
                    f"below prior high Rs{lifetime_high:.2f} "
                    f"({lth_exclude}-{lth_lookback}d window)",
                    "info",
                )
    ema_period  = int(cfg.get("ema_period", 9))
    ema_enabled = cfg.get("ema_filter_enabled", True)
    ema_values  = compute_ema(closes, ema_period)

    # ── CONDITIONS 1-4: Scan backwards for breakdown candle ───────────────────
    # The breakdown candle and the EMA breach do NOT have to occur on the same
    # day. We look back up to `surge_recency_days` candles to find a candle
    # that:
    #   (a) closed below the prior candle's low by >= min_drop_percent    [C1+C2]
    #   (b) had volume >= min_breakdown_volume_ratio x 20d avg            [C3]
    # Once a valid breakdown candle is found, we check whether EMA was
    # breached on THAT candle OR on any candle between it and today.       [C4]
    #
    # Example — CDSL 21-22 Apr 2026:
    #   21 Apr: closes below prev low + high volume  ← breakdown candle (Gate 1)
    #   22 Apr: closes below 9-EMA with volume       ← Gate 2 confirmed separately
    #   Today:  price rallies back → sell zone fires  ← Gate 3 proximity
    #
    # Gate 2 (EMA+volume after breakdown) is tracked per-signal by
    # update_sell_zone_gates() and is NOT required at signal creation time.

    min_drop      = cfg.get("min_drop_percent", 0.5)
    min_vol_ratio = cfg.get("min_breakdown_volume_ratio", 1.2)
    lookback_bd   = int(cfg.get("surge_recency_days", 5))  # how far back to search

    breakdown_idx    = None   # index of the confirmed breakdown candle
    today_close      = float(closes[-1])
    yesterday_high   = None
    yesterday_low    = None
    yesterday_close  = None
    drop_pct         = None
    breakdown_vol    = None
    avg_vol_20d      = None
    volume_ratio     = None
    ema_at_detection = None
    below_ema        = False

    # Search from most recent candle backwards (index -1 = today, -2 = yesterday …)
    for offset in range(1, min(lookback_bd + 2, len(closes))):
        bd_idx    = len(closes) - offset          # candidate breakdown candle
        prev_idx  = bd_idx - 1                    # candle before it

        if prev_idx < 1:
            break

        bd_close  = float(closes[bd_idx])
        prev_low  = float(lows[prev_idx])
        prev_high = float(highs[prev_idx])
        prev_close= float(closes[prev_idx])

        # C1: must close below prior candle's low
        if bd_close >= prev_low:
            continue

        # C2: drop must be meaningful
        _drop = (prev_low - bd_close) / prev_low * 100
        if _drop < min_drop:
            continue

        # C3: volume check
        _bd_vol      = float(vols[bd_idx]) if vols[bd_idx] > 0 else 0.0
        _vol_window  = vols[max(0, bd_idx - 20):bd_idx]
        _avg_vol     = float(_vol_window.mean()) if len(_vol_window) > 0 else 0.0
        _vol_ratio   = (_bd_vol / _avg_vol) if _avg_vol > 0 else 0.0

        if _vol_ratio < min_vol_ratio:
            continue

        # EMA value at breakdown for display/reference (Gate 2 EMA+volume check
        # is handled separately by update_sell_zone_gates after signal creation).
        _ema_at = float(ema_values[bd_idx])

        # ── Valid breakdown found ─────────────────────────────────────────────
        breakdown_idx    = bd_idx
        yesterday_high   = prev_high
        yesterday_low    = prev_low
        yesterday_close  = prev_close
        drop_pct         = _drop
        breakdown_vol    = _bd_vol
        avg_vol_20d      = _avg_vol
        volume_ratio     = _vol_ratio
        ema_at_detection = _ema_at
        below_ema        = bool(closes[bd_idx] < ema_values[bd_idx])

        if job_id:
            bd_date = dates[bd_idx].strftime("%Y-%m-%d")
            job_log(job_id,
                f"{ticker} breakdown candle: {bd_date}  "
                f"close Rs{closes[bd_idx]:.2f}  prev_low Rs{prev_low:.2f}  "
                f"drop={drop_pct:.2f}%  vol={volume_ratio:.2f}x",
                "info")
        break   # use the most recent valid breakdown

    if breakdown_idx is None:
        if job_id:
            job_log(job_id,
                f"{ticker} no valid breakdown candle in last {lookback_bd} sessions",
                "fail")
        return None

    # ── CONDITION 5: Continuous surge ending recently ──────────────────────
    min_gain     = cfg["min_gain_percent"]
    min_green    = cfg["min_green_candles"]
    recency_days = int(cfg.get("surge_recency_days", 5))

    # Operate on candles BEFORE the breakdown candle (exclude breakdown onward).
    # breakdown_idx points to the breakdown candle; surge must end before it.
    scan_closes = closes[:breakdown_idx]
    scan_opens  = opens[:breakdown_idx]
    scan_lows   = lows[:breakdown_idx]
    scan_highs  = highs[:breakdown_idx]   # pre-breakdown highs only (excludes live-price injection)
    scan_dates  = dates[:breakdown_idx]
    n           = len(scan_closes)

    if n < min_green + 2:
        return None

    # Surge end must be within recency_days sessions BEFORE the breakdown candle
    min_end_idx  = n - recency_days
    window_min   = max(min_green + 1, 3)

    best_gain   = 0.0
    best_greens = 0
    best_reds   = 0
    best_window = None

    for wsize in range(window_min, n + 1):
        for start in range(0, n - wsize + 1):
            end = start + wsize - 1

            # Enforce recency
            if end < min_end_idx:
                continue

            net_gain = ((scan_closes[end] - scan_closes[start]) / scan_closes[start]) * 100
            if net_gain < min_gain:
                continue

            # Continuity check: red candles allowed only if higher-low holds
            is_cont, red_count, green_count = check_surge_continuity(
                scan_closes, scan_opens, scan_lows, start, end
            )
            if not is_cont or green_count < min_green:
                continue

            # Keep best by gain; prefer more recent end on tie
            prev_end = best_window[1] if best_window else -1
            if net_gain > best_gain or (net_gain == best_gain and end > prev_end):
                best_gain   = net_gain
                best_greens = green_count
                best_reds   = red_count
                best_window = (start, end)

    if best_window is None or best_gain < min_gain:
        if job_id:
            job_log(job_id,
                f"{ticker} no continuous surge >={min_gain}% ending within "
                f"last {recency_days} sessions",
                "fail")
        return None

    surge_start_idx, surge_end_idx = best_window
    surge_end_date  = scan_dates[surge_end_idx]
    days_since_end  = (datetime.now(IST).date() - surge_end_date.date()).days

    # ── CRITICAL: Use SURGE_HIGH as resistance (not yesterday_high) ────────
    # The surge window's highest point is the true resistance level where
    # price exhausted momentum. Even after breakdown + recovery, this level
    # will likely reject price again (as seen in DIXON 20 May 2025 example).
    surge_window_highs = scan_highs[surge_start_idx:surge_end_idx + 1]   # ✅ uses pre-breakdown slice — excludes live-price injection on today's candle
    surge_high         = float(np.max(surge_window_highs))

    if job_id:
        job_log(job_id,
            f"{ticker} PASS  surge={best_gain:.1f}%  greens={best_greens}  "
            f"allowed_reds={best_reds}  ended {days_since_end}d ago  "
            f"drop={drop_pct:.2f}%  vol={volume_ratio:.2f}x  "
            f"EMA{ema_period}={ema_at_detection:.2f} (breached at detection)  "
            f"surge_high=Rs{surge_high:.2f} (resistance)  "
            f"yesterday_high=Rs{yesterday_high:.2f}",
            "pass")

    return {
        "ticker":                  ticker,
        "today_close":             round(today_close, 2),
        "surge_high":              round(surge_high, 2),        # TRUE RESISTANCE (surge peak)
        "yesterday_high":          round(yesterday_high, 2),    # Day before breakdown
        "yesterday_close":         round(yesterday_close, 2),
        "yesterday_low":           round(yesterday_low, 2),
        "drop_pct":                round(drop_pct, 2),
        "breakdown_volume":        int(breakdown_vol),
        "avg_volume_20d":          int(avg_vol_20d),
        "volume_ratio":            round(volume_ratio, 2),
        "ema_at_detection":        round(ema_at_detection, 2),  # EMA value at breakdown candle
        "ema_breached_at_detection": True,                      # Legacy compat
        "below_ema":               below_ema,                   # kept for UI/legacy compat
        "surge_gain_pct":          round(best_gain, 2),
        "surge_candles":           best_greens,
        "surge_allowed_reds":      best_reds,
        "surge_start_date":        scan_dates[surge_start_idx].strftime("%Y-%m-%d"),
        "surge_end_date":          surge_end_date.strftime("%Y-%m-%d"),
        "days_since_surge_end":    days_since_end,
        "breakdown_date":          dates[breakdown_idx].strftime("%Y-%m-%d"),
        # Previous-Pullup-High filter metadata (Check 1)
        "prev_pullup_high":        round(lifetime_high, 2) if lifetime_high else None,
    }


# ─────────────────────────────────────────────────────────────
#  SCREENING -- STEP 2
# ─────────────────────────────────────────────────────────────
def find_best_strike(candidate: Dict, cfg: Dict, job_id: str = "") -> Optional[Dict]:
    ticker      = candidate["ticker"]
    surge_high  = candidate["surge_high"]   # TRUE resistance (surge peak)

    if job_id:
        job_log(job_id, f"{ticker} -> fetching option chain...", "info")

    strike_map, expiry = fetch_option_chain(ticker)
    if strike_map is None or not strike_map:
        if job_id:
            job_log(job_id, f"{ticker} option chain empty", "fail")
        return None

    # Strike anchored ABOVE surge_high (true resistance), not yesterday_high.
    # CE goes ITM only if price breaks above the surge peak -- exactly
    # the scenario our thesis says is unlikely (as seen in DIXON 20 May example).
    strike = nearest_round_strike_above(surge_high, strike_map)
    if strike is None:
        if job_id:
            job_log(job_id,
                f"{ticker} no liquid strike above surge_high Rs{surge_high:.2f}",
                "fail")
        return None

    ce_data = strike_map[strike]
    ce_ltp  = ce_data["CE_ltp"]
    ce_oi   = ce_data["CE_oi"]

    if job_id:
        job_log(job_id,
            f"{ticker} -> strike Rs{strike:.0f} CE (surge_high Rs{surge_high:.2f}) "
            f"| LTP Rs{ce_ltp:.2f} | OI {ce_oi:,}",
            "info")

    if ce_ltp <= 0:
        if job_id:
            job_log(job_id, f"{ticker} strike Rs{strike:.0f} CE LTP=0 (illiquid)", "fail")
        return None

    ce_hist_high        = None
    ce_above_30d_status = "Unverified"
    nse_hist_calls      = 0
    remark              = "Unverified 30d High"

    # ── CE historical-high check: deferred during bulk scan ──────────────────
    # During a scan of 200+ tickers this check adds 1–3s per candidate (one
    # extra NSE HTTP call + sleep).  We skip it here and mark CE_Hist_Checked=False.
    # The check runs on-demand when the user opens a signal's detail modal
    # (POST /api/signals/{ticker}/ce-check).  If ce_above_historical_high is
    # disabled in settings the check never runs at all.
    ce_filter_enabled = cfg.get("ce_above_historical_high", True)
    if not ce_filter_enabled:
        remark = "check disabled"
    # else: remains "Unverified 30d High" — deferred

    if job_id:
        job_log(job_id,
            f"{ticker} SIGNAL  Rs{strike:.0f} CE  LTP Rs{ce_ltp:.2f}  [{remark}]",
            "signal")

    # ── CHECK 2: Bear Call Spread ─────────────────────────────────────────────
    # Strategy: SELL short CE (at `strike`, above surge high)  +  BUY hedge CE
    # (N intervals higher) to cap max loss.
    #
    #   Net Premium (credit)  = short_CE_ltp  - hedge_CE_ltp
    #   Max Profit            = Net Premium   (price stays below short strike)
    #   Max Loss              = spread_width  - Net Premium
    #   Breakeven             = short_strike  + Net Premium
    #   Risk/Reward           = Max Loss / Net Premium
    #
    # The hedge leg is sourced directly from the same option-chain snapshot
    # already in memory — no extra NSE call needed.
    # ─────────────────────────────────────────────────────────────────────────
    spread_intervals    = int(cfg.get("bear_call_spread_width_intervals", 1))
    hedge_strike        = None
    hedge_ce_ltp        = 0.0
    hedge_ce_oi         = 0
    spread_net_premium  = None
    spread_max_profit   = None
    spread_max_loss     = None
    spread_breakeven    = None
    spread_risk_reward  = None
    spread_width        = None

    if spread_intervals > 0:
        interval   = nse_strike_interval(strike)
        # Try spread_intervals, then spread_intervals+1, then +2 to find a
        # liquid hedge leg in the strike_map.
        for extra in range(spread_intervals, spread_intervals + 3):
            candidate_hedge = strike + extra * interval
            if candidate_hedge in strike_map and strike_map[candidate_hedge]["CE_ltp"] > 0:
                hedge_strike  = candidate_hedge
                hedge_ce_ltp  = float(strike_map[candidate_hedge]["CE_ltp"])
                hedge_ce_oi   = int(strike_map[candidate_hedge]["CE_oi"])
                break

        if hedge_strike is not None and hedge_ce_ltp > 0 and ce_ltp > hedge_ce_ltp:
            spread_width       = round(hedge_strike - strike, 2)
            spread_net_premium = round(ce_ltp - hedge_ce_ltp, 2)
            spread_max_profit  = spread_net_premium                              # per unit
            spread_max_loss    = round(spread_width - spread_net_premium, 2)    # per unit
            spread_breakeven   = round(strike + spread_net_premium, 2)
            spread_risk_reward = (
                round(spread_max_loss / spread_net_premium, 2)
                if spread_net_premium > 0 else None
            )
            if job_id:
                job_log(
                    job_id,
                    f"{ticker} Bear-Call-Spread: "
                    f"SELL Rs{strike:.0f} CE @ Rs{ce_ltp:.2f}  "
                    f"BUY Rs{hedge_strike:.0f} CE @ Rs{hedge_ce_ltp:.2f}  "
                    f"| Net Credit Rs{spread_net_premium:.2f}  "
                    f"Max-Profit Rs{spread_max_profit:.2f}  "
                    f"Max-Loss Rs{spread_max_loss:.2f}  "
                    f"Breakeven Rs{spread_breakeven:.2f}  "
                    f"R:R 1:{spread_risk_reward}",
                    "signal",
                )
        else:
            if job_id:
                reason = (
                    "no liquid hedge strike found above short CE"
                    if hedge_strike is None
                    else f"hedge CE LTP Rs{hedge_ce_ltp:.2f} >= short CE LTP Rs{ce_ltp:.2f} (debit spread — skip)"
                )
                job_log(job_id, f"{ticker} Bear-Call-Spread: {reason}", "warn")

    return {
        "Ticker":               ticker,
        # Price action
        "Today_Close":          candidate["today_close"],
        "Surge_High":           candidate["surge_high"],        # TRUE RESISTANCE (surge peak)
        "Yesterday_High":       candidate["yesterday_high"],    # Day before breakdown
        "Yesterday_Close":      candidate["yesterday_close"],
        "Yesterday_Low":        candidate["yesterday_low"],
        "Drop_Pct":             candidate["drop_pct"],
        # Volume
        "Breakdown_Volume":     candidate["breakdown_volume"],
        "Avg_Volume_20d":       candidate["avg_volume_20d"],
        "Volume_Ratio":         candidate["volume_ratio"],
        # Trend — EMA was breached at initial detection; not re-checked on retest days
        "EMA_At_Detection":         candidate["ema_at_detection"],
        "EMA_Breached_At_Detection": candidate["ema_breached_at_detection"],
        "Below_EMA":                candidate["below_ema"],          # legacy compat
        # Surge
        "Surge_Gain_Pct":       candidate["surge_gain_pct"],
        "Surge_Candles":        candidate["surge_candles"],
        "Surge_Allowed_Reds":   candidate["surge_allowed_reds"],
        "Surge_Start":          candidate["surge_start_date"],
        "Surge_End":            candidate["surge_end_date"],
        "Days_Since_Surge_End": candidate["days_since_surge_end"],
        # Previous-Pullup-High filter metadata (Check 1)
        "Prev_Pullup_High":     candidate.get("prev_pullup_high"),
        # Short CE leg  (Check 2 — primary sell leg)
        "Suggested_Strike":     strike,
        "Expiry":               expiry,
        "CE_LTP":               round(ce_ltp, 2),
        "CE_OI":                ce_oi,
        "CE_30d_High":          round(ce_hist_high, 2) if ce_hist_high else "N/A",
        "CE_Above_30d_High":    ce_above_30d_status,
        "Remark":               remark,
        "NSE_Hist_Calls":       nse_hist_calls,
        "CE_Hist_Checked":      False,   # set True after on-demand check
        # Bear Call Spread — hedge (buy) leg  (Check 2)
        # All None when spread disabled or no liquid hedge found.
        "BCS_Hedge_Strike":     hedge_strike,
        "BCS_Hedge_CE_LTP":     round(hedge_ce_ltp, 2) if hedge_ce_ltp else None,
        "BCS_Hedge_CE_OI":      hedge_ce_oi if hedge_ce_oi else None,
        "BCS_Spread_Width":     spread_width,
        "BCS_Net_Premium":      spread_net_premium,
        "BCS_Max_Profit":       spread_max_profit,
        "BCS_Max_Loss":         spread_max_loss,
        "BCS_Breakeven":        spread_breakeven,
        "BCS_Risk_Reward":      spread_risk_reward,
        # Meta
        "Status":               "Momentum Lost",
        "Sell_Alert_Sent":      False,
        # Gate 2 — EMA+volume breach after breakdown (sticky)
        "Breakdown_Date":            candidate.get("breakdown_date"),
        "Sell_Zone_EMA_Confirmed":   False,
        "Sell_Zone_EMA_Confirm_Date": None,
        "Scanned_At":           now_ist_str(),
    }


# ─────────────────────────────────────────────────────────────
#  GATE 2 — EMA + VOLUME CONFIRMATION AFTER BREAKDOWN
#
#  After the momentum break (Gate 1 / signal creation), price must
#  close below the 9-EMA at least ONCE with above-average volume
#  before the sell zone (Gate 3) can fire.
#
#  This is tracked per-signal as Sell_Zone_EMA_Confirmed (sticky).
# ─────────────────────────────────────────────────────────────
def check_ema_confirmed_after_breakdown(
    ticker: str, breakdown_date_str: str, cfg: Dict
) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Scan from breakdown_date forward.  Return
        (True, confirm_date, recomputed_surge_high)
    on the first candle where close < 9-EMA AND volume >= min_breakdown_volume_ratio × 20d avg.

    recomputed_surge_high mirrors the backtest.py Gate-2 refinement:
        After Gate 2 fires, the true resistance is the highest HIGH in the
        surge_recency_days window BEFORE the breakdown candle, up to (but not
        including) the Gate-2 candle.  The original Gate-1 surge detection
        sometimes finds a small late sub-window and misses the real price peak;
        this broader max() always captures it correctly.
    """
    ema_period    = int(cfg.get("ema_period", 9))
    min_vol_ratio = float(cfg.get("min_breakdown_volume_ratio", 0.5))
    recency_days  = int(cfg.get("surge_recency_days", 5))

    hist = get_price_history(ticker, cfg.get("lookback_days", 45))
    if hist is None or hist.empty:
        return False, None, None

    hist = hist.sort_index()
    closes = hist["Close"].values
    highs  = hist["High"].values          # needed for surge_high recomputation
    vols   = hist["Volume"].values
    dates  = hist.index

    ema_values = compute_ema(closes, ema_period)

    try:
        bd_date = datetime.strptime(breakdown_date_str, "%Y-%m-%d").date()
    except Exception:
        return False, None, None

    # 20d average volume over the full window (pre-breakdown baseline)
    avg_vol_20d = float(np.mean(vols[-20:])) if len(vols) >= 20 else float(np.mean(vols))
    if avg_vol_20d <= 0:
        return False, None, None

    # Find start index — first candle at or after the breakdown date
    start_idx = next(
        (i for i, d in enumerate(dates) if d.date() >= bd_date),
        None
    )
    if start_idx is None:
        return False, None, None

    for i in range(start_idx, len(closes)):
        if closes[i] < ema_values[i]:
            vol_ratio = float(vols[i]) / avg_vol_20d
            if vol_ratio >= min_vol_ratio:
                # ── Recompute surge_high: max high from (breakdown - recency_days)
                #    up to (but not including) this Gate-2 candle.
                #    Mirrors backtest.py backtest_ticker() lines 592-596.
                lookback_start = max(0, start_idx - recency_days)
                if lookback_start < i:
                    recomputed_surge_high = float(np.max(highs[lookback_start:i]))
                else:
                    recomputed_surge_high = None
                return True, dates[i].strftime("%Y-%m-%d"), recomputed_surge_high

    return False, None, None


def update_sell_zone_gates(cfg: Dict) -> int:
    """
    Called every scan cycle.  For each signal where Gate 2 is not yet
    confirmed, check if price has now closed below 9-EMA with volume.
    Gate 2 confirmation is STICKY — once set it is never cleared.

    When Gate 2 fires, surge_high is RECOMPUTED to the max high over
    surge_recency_days before breakdown up to the Gate-2 candle (mirrors
    backtest.py).  If the recomputed value is higher than the stored one,
    Suggested_Strike is also updated using NSE standard intervals (no
    extra option-chain call needed).

    Returns the count of newly confirmed signals.
    """
    signals = load_signals()
    newly_confirmed = 0
    changed = False

    for sig in signals:
        if sig.get("Sell_Zone_EMA_Confirmed"):
            continue  # already confirmed — skip

        breakdown_date = sig.get("Breakdown_Date")
        if not breakdown_date:
            continue

        ticker = sig["Ticker"]
        confirmed, confirm_date, new_surge_high = check_ema_confirmed_after_breakdown(
            ticker, breakdown_date, cfg
        )
        time.sleep(random.uniform(0.3, 0.6))  # throttle yfinance calls

        if confirmed:
            sig["Sell_Zone_EMA_Confirmed"]    = True
            sig["Sell_Zone_EMA_Confirm_Date"] = confirm_date
            newly_confirmed += 1
            changed = True
            print(f"[Gate2] {ticker} Gate 2 confirmed on {confirm_date}")

            # ── Surge-high refinement (backtest.py parity) ────────────────
            # The Gate-1 surge detection may have found a narrow late window,
            # missing the actual price peak.  The broader max() from
            # check_ema_confirmed_after_breakdown() corrects this.
            if new_surge_high is not None and new_surge_high > 0:
                old_surge_high = float(sig.get("Surge_High") or 0)
                if new_surge_high != old_surge_high:
                    sig["Surge_High"] = round(new_surge_high, 2)
                    print(f"[Gate2] {ticker} surge_high refined: "
                          f"Rs{old_surge_high:.2f} → Rs{new_surge_high:.2f}")

                    # Recompute strike if the resistance ceiling moved up.
                    # Use math-only helper to avoid an extra NSE call.
                    if new_surge_high > old_surge_high:
                        new_strike = nearest_otm_strike_math(new_surge_high)
                        old_strike = sig.get("Suggested_Strike", 0)
                        if new_strike != old_strike:
                            sig["Suggested_Strike"] = new_strike
                            print(f"[Gate2] {ticker} strike updated: "
                                  f"{old_strike} → {new_strike:.0f}")

    if changed:
        save_signals(signals)

    return newly_confirmed



#  Reference: Yesterday_High = resistance ceiling defined at scan time.
#
#  Alert fires when:
#      floor = yesterday_high * (1 - proximity_pct / 100)
#      floor <= cur_price <= yesterday_high
#
#  Price above yesterday_high = breakout, thesis invalidated, skip.
#  Duplicate alerts for the same ticker within 30 min are suppressed.
# ─────────────────────────────────────────────────────────────
def check_proximity_alerts(signals: List[Dict], cfg: Dict) -> List[Dict]:
    proximity_pct  = cfg["price_proximity_percent"]
    sell_zone_hits = []

    for sig in signals:
        ticker = sig["Ticker"]

        # ── GATE 2 GUARD: EMA+volume breach after breakdown must be confirmed ─
        if not sig.get("Sell_Zone_EMA_Confirmed"):
            time.sleep(random.uniform(0.1, 0.3))
            continue

        # Surge_High (surge peak) is the primary resistance ceiling.
        resistance = (
            sig.get("Surge_High")
            or sig.get("Yesterday_High")
            or sig.get("Ten_Day_High")
            or sig.get("Suggested_Strike")
        )
        if not resistance:
            time.sleep(random.uniform(0.2, 0.5))
            continue

        cur_price = get_current_price(ticker)
        if cur_price is None:
            time.sleep(random.uniform(0.2, 0.5))
            continue

        # Price broke above surge_high → thesis invalidated (breakout confirmed)
        if cur_price > resistance:
            time.sleep(random.uniform(0.2, 0.5))
            continue

        dist_pct = (resistance - cur_price) / resistance * 100

        if dist_pct <= proximity_pct:
            g2_date = sig.get("Sell_Zone_EMA_Confirm_Date", "?")
            hit = {
                **sig,
                "Current_Price":          round(cur_price, 2),
                "Resistance_High":        round(resistance, 2),
                "Distance_From_High_Pct": round(dist_pct, 2),
                "Alert_Time":             now_ist_str(),
            }
            sell_zone_hits.append(hit)
            send_telegram(
                cfg["telegram_bot_token"],
                cfg["telegram_chat_id"],
                f"*SELL ZONE — {ticker}* ✅ All 3 Gates Passed\n"
                f"Gate 1 ✅ Momentum break (close < prev low)\n"
                f"Gate 2 ✅ EMA breach confirmed {g2_date}\n"
                f"Gate 3 ✅ Price Rs{cur_price:.2f} within "
                f"{dist_pct:.2f}% of Surge High Rs{resistance:.2f} "
                f"(limit {proximity_pct}%)\n"
                f"Strike Rs{sig.get('Suggested_Strike','?')} CE  "
                f"Expiry {sig.get('Expiry','?')}"
            )

        time.sleep(random.uniform(0.2, 0.5))

    return sell_zone_hits




# ─────────────────────────────────────────────────────────────
#  SIGNAL PRUNING
# ─────────────────────────────────────────────────────────────
def prune_stale_signals(signals: List[Dict], cfg: Dict) -> Tuple[List[Dict], int]:
    max_age = int(cfg.get("max_signal_age_days", 5))
    cutoff  = datetime.now(IST) - timedelta(days=max_age)
    fresh, pruned = [], 0
    for s in signals:
        try:
            scanned = datetime.strptime(s["Scanned_At"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
            if scanned >= cutoff:
                fresh.append(s)
            else:
                pruned += 1
        except Exception:
            fresh.append(s)
    return fresh, pruned


# ─────────────────────────────────────────────────────────────
#  SCREEN JOB
# ─────────────────────────────────────────────────────────────
def run_screen_job(tickers: List[str], cfg: Dict, job_id: str):
    """
    Optimised scan pipeline for 200+ tickers.

    Phase 1 — Parallel yfinance prefetch (ThreadPoolExecutor).
              All tickers fetched concurrently; results stored in _yf_cache.
              yfinance/Yahoo has no meaningful rate limit for equity history.

    Phase 2 — Parallel surge detection (CPU + cache reads, no IO).
              check_surge_and_loss() is now pure-memory after Phase 1.
              Live price (fast_info) calls are also batched here.

    Phase 3 — Sequential NSE option-chain calls with polite jitter.
              NSE session is shared and rate-sensitive; we keep one call
              at a time with the existing sleep pattern.  This is where
              most of the remaining elapsed time goes for candidates only
              (~5–15% of tickers typically qualify as candidates).
    """
    global _yf_cache
    total     = len(tickers)
    signals   = []
    nse_calls = 0

    job_log(job_id, f"Starting scan of {total} tickers (parallel prefetch)…", "info")

    # ── Reset per-scan cache ─────────────────────────────────────────────────
    with _yf_cache_lock:
        _yf_cache = {}

    # ── PHASE 1: Parallel yfinance prefetch ─────────────────────────────────
    # Fetch up to 16 tickers concurrently.  yfinance handles its own HTTP
    # pooling; 16 workers is a sweet spot — more workers don't help much
    # because each call is a single HTTP round-trip (~0.5s).
    PREFETCH_WORKERS = min(16, total)
    job_log(job_id, f"Phase 1 — prefetching {total} tickers ({PREFETCH_WORKERS} workers)…", "info")

    prefetch_done = 0
    with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
        future_to_ticker = {pool.submit(_fetch_and_cache, t): t for t in tickers}
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            prefetch_done += 1
            try:
                result = future.result()
                if result is None:
                    job_log(job_id, f"{t} — no price history", "fail")
            except Exception as exc:
                job_log(job_id, f"{t} prefetch error: {exc}", "fail")
            job_progress(job_id, prefetch_done, total, t)

    job_log(job_id, f"Phase 1 complete — {len(_yf_cache)}/{total} tickers cached", "info")

    # ── PHASE 2: Parallel surge detection ───────────────────────────────────
    # All data is now in _yf_cache; check_surge_and_loss() only reads memory.
    # Live price calls (fast_info) are lightweight but still IO — batch at 8.
    SURGE_WORKERS = min(8, len(_yf_cache))
    job_log(job_id, f"Phase 2 — surge detection ({SURGE_WORKERS} workers)…", "info")

    candidates: List[Dict] = []
    surge_done  = 0
    cached_tickers = list(_yf_cache.keys())

    with ThreadPoolExecutor(max_workers=SURGE_WORKERS) as pool:
        future_to_ticker = {
            pool.submit(check_surge_and_loss, t, cfg, job_id): t
            for t in cached_tickers
        }
        for future in as_completed(future_to_ticker):
            t = future_to_ticker[future]
            surge_done += 1
            try:
                candidate = future.result()
                if candidate is not None:
                    candidates.append(candidate)
            except Exception as exc:
                job_log(job_id, f"{t} surge-check error: {exc}", "fail")
            job_progress(job_id, total + surge_done, total * 2, t)

    job_log(job_id,
        f"Phase 2 complete — {len(candidates)} candidate(s) from {len(cached_tickers)} cached tickers",
        "info")

    # ── PHASE 3: Sequential NSE option-chain calls (rate-limit safe) ─────────
    # NSE uses a shared session with cookies; concurrent requests get 403/429.
    # Keep one call at a time with polite jitter — same as before.
    job_log(job_id, f"Phase 3 — NSE option chain for {len(candidates)} candidate(s)…", "info")

    for idx, candidate in enumerate(candidates):
        ticker = candidate["ticker"]
        job_progress(job_id, total * 2 + idx, total * 2 + len(candidates), ticker)
        _nse_sleep(1.5, 2.5)   # adaptive: scales if NSE starts rate-limiting
        signal = find_best_strike(candidate, cfg, job_id)
        nse_calls += 2
        if signal is None:
            time.sleep(random.uniform(0.5, 1.0))
            continue
        signals.append(signal)
        _nse_sleep(1.0, 2.0)

    # ── Gate 2, prune, merge ─────────────────────────────────────────────────
    gate2_new = update_sell_zone_gates(cfg)
    if gate2_new:
        job_log(job_id, f"Gate 2 newly confirmed for {gate2_new} signal(s)", "info")

    existing, pruned = prune_stale_signals(load_signals(), cfg)
    if pruned:
        job_log(job_id, f"Pruned {pruned} stale signal(s) (>{cfg.get('max_signal_age_days',5)}d)", "info")

    seen  = {s["Ticker"] for s in existing}
    added = 0
    for s in signals:
        if s["Ticker"] not in seen:
            existing.append(s)
            seen.add(s["Ticker"])
            added += 1
        else:
            old = next((ex for ex in existing if ex["Ticker"] == s["Ticker"]), None)
            if old:
                s["Sell_Zone_EMA_Confirmed"]    = old.get("Sell_Zone_EMA_Confirmed", False)
                s["Sell_Zone_EMA_Confirm_Date"] = old.get("Sell_Zone_EMA_Confirm_Date")
            existing = [s if ex["Ticker"] == s["Ticker"] else ex for ex in existing]

    save_signals(existing)

    log = load_scan_log()
    log.append({
        "time":            now_ist_str(),
        "tickers_scanned": total,
        "signals_found":   len(signals),
        "nse_calls":       nse_calls,
    })
    save_scan_log(log[-100:])

    job_log(job_id,
        f"Done — {len(signals)} signal(s) found, {added} new, {pruned} pruned.", "info")
    job_progress(job_id, total, total, "")
    job_done(job_id, signals)

    # Clear cache after job to free memory
    with _yf_cache_lock:
        _yf_cache = {}


# ─────────────────────────────────────────────────────────────
#  PREMIUM SELL ZONE SCAN
#
#  For every Gate-2-confirmed signal that is currently in the
#  proximity range (Gate 3), this scan:
#    1. Measures live volume vs 20d average (volume surge ratio).
#    2. Fetches option chain and builds an ATM + OTM ladder
#       showing CE premiums across multiple strikes.
#    3. Computes a "Premium Score" = vol_surge × ATM_CE_LTP.
#       Higher score ↔ more premium available right now.
#
#  Thesis: high volume at resistance + elevated CE premiums =
#  ideal time to SELL CE for maximum theta decay profit.
# ─────────────────────────────────────────────────────────────
def scan_premium_zone_job(cfg: Dict, job_id: str) -> List[Dict]:
    """
    Scans Gate-2-confirmed signals that are in proximity range.
    Returns list of dicts sorted by Premium_Score descending.
    """
    signals       = load_signals()
    prox_pct      = float(cfg.get("price_proximity_percent", 2.0))
    min_vol_surge = float(cfg.get("premium_min_vol_surge", 1.5))
    otm_depth     = int(cfg.get("premium_otm_depth", 3))

    # Only Gate-2 confirmed signals are considered — breakdown + EMA breach validated
    gate2 = [s for s in signals if s.get("Sell_Zone_EMA_Confirmed")]
    job_log(job_id,
        f"Premium Zone: {len(gate2)} Gate-2-confirmed signal(s) to evaluate",
        "info")

    results: List[Dict] = []
    skipped_low_vol: int   = 0   # Gate-2+proximity stocks skipped due to vol < threshold

    for sig in gate2:
        ticker     = sig["Ticker"]
        resistance = (
            sig.get("Surge_High")
            or sig.get("Yesterday_High")
            or sig.get("Suggested_Strike")
        )
        if not resistance:
            continue

        # ── Live price ──────────────────────────────────────────────────────
        cur_price = get_current_price(ticker)
        if cur_price is None:
            job_log(job_id, f"{ticker} price unavailable — skipping", "warn")
            time.sleep(random.uniform(0.2, 0.5))
            continue

        # Thesis invalidated if price already broke above resistance
        if cur_price > resistance:
            job_log(job_id,
                f"{ticker} price Rs{cur_price:.2f} > resistance Rs{resistance:.2f} (breakout) — skip",
                "fail")
            time.sleep(random.uniform(0.2, 0.4))
            continue

        dist_pct = (resistance - cur_price) / resistance * 100

        # Gate 3: must be within proximity_percent of resistance
        if dist_pct > prox_pct:
            job_log(job_id,
                f"{ticker} dist {dist_pct:.1f}% > {prox_pct}% proximity limit — skip",
                "fail")
            time.sleep(random.uniform(0.2, 0.4))
            continue

        job_log(job_id,
            f"{ticker} ✓ in proximity {dist_pct:.1f}%  Rs{cur_price:.2f} → Rs{resistance:.2f}  "
            f"checking volume + options...",
            "info")

        # ── Volume surge (must coincide with price TESTING resistance) ──────────
        # KEY FIX: volume is only counted as a "resistance-zone surge" when
        # today's HIGH is also within proximity of resistance.  This prevents
        # falsely flagging high-volume sessions that happened far from resistance
        # (e.g. gap-down open or early morning dump) but where LTP drifted near
        # resistance by end of day.
        vol_surge_ratio      = None
        today_vol            = None
        avg_vol_20d          = None
        today_high           = None
        high_near_resistance = False   # True iff today's HIGH tested resistance zone
        high_dist_pct        = None

        hist = get_price_history(ticker, 25)
        if hist is not None and len(hist) >= 5:
            vols       = hist["Volume"].values
            highs_hist = hist["High"].values
            # 20d average using candles BEFORE today
            window = vols[-21:-1] if len(vols) >= 22 else vols[:-1]
            avg_vol_20d = float(window.mean()) if len(window) > 0 else None
            today_vol   = float(vols[-1])
            today_high  = float(highs_hist[-1])
            if avg_vol_20d and avg_vol_20d > 0:
                vol_surge_ratio = round(today_vol / avg_vol_20d, 2)

            # Check whether today's HIGH reached the resistance zone.
            # - If high < resistance: distance must be within prox_pct
            # - If high > resistance: intraday wick pierced/rejected at resistance
            #   (still counts — price definitely tested that level today)
            if today_high is not None:
                if today_high <= resistance:
                    high_dist_pct        = round((resistance - today_high) / resistance * 100, 2)
                    high_near_resistance = high_dist_pct <= prox_pct
                else:
                    # Intraday candle wick exceeded resistance then came back → strong rejection
                    high_dist_pct        = 0.0
                    high_near_resistance = True

        vol_label = f"{vol_surge_ratio:.2f}x" if vol_surge_ratio is not None else "N/A"

        # Both conditions must hold:
        #   1. Volume is elevated vs 20d average
        #   2. Today's HIGH confirms price actually reached the resistance zone
        is_vol_surge = (
            vol_surge_ratio is not None
            and vol_surge_ratio >= min_vol_surge
            and high_near_resistance
        )

        high_label = (
            f"Rs{today_high:.2f} ({high_dist_pct:.1f}% from resistance)"
            if today_high is not None else "N/A"
        )
        vol_reason = (
            "✓ HIGH VOL AT RESISTANCE"    if is_vol_surge else
            f"⚠ vol={vol_label} OK but HIGH Rs{today_high:.2f} not near resistance ({high_dist_pct:.1f}% > {prox_pct}%)"
                if (vol_surge_ratio is not None and vol_surge_ratio >= min_vol_surge and not high_near_resistance) else
            f"⚠ vol={vol_label} below threshold {min_vol_surge}x"
        )
        job_log(job_id,
            f"{ticker} volume @ resistance: {vol_label}  today_high={high_label}  {vol_reason}",
            "info" if is_vol_surge else "warn")

        # ── HARD GATE: skip if vol surge not confirmed at resistance ──────────
        # Both conditions must be true before we bother fetching option chain:
        #   • vol_surge_ratio >= min_vol_surge  (e.g. 1.5x)
        #   • today_high within prox_pct of resistance  (price actually tested it)
        # A 0.23x day is NOT a premium zone — CE premiums are stale/undemanded.
        if not is_vol_surge:
            job_log(job_id,
                f"{ticker} SKIP — vol {vol_label} not a volume surge at resistance "                f"(threshold {min_vol_surge}x, high_near_resistance={high_near_resistance})",
                "fail")
            skipped_low_vol += 1
            time.sleep(random.uniform(0.2, 0.4))
            continue

        time.sleep(random.uniform(2.0, 3.5))

        # ── Option chain ─────────────────────────────────────────────────────
        strike_map, expiry = fetch_option_chain(ticker)
        if not strike_map:
            job_log(job_id, f"{ticker} option chain unavailable", "fail")
            continue

        sorted_strikes = sorted(strike_map.keys())

        # ATM = strike closest to current price
        atm_strike = min(sorted_strikes, key=lambda k: abs(k - cur_price))
        atm_idx    = sorted_strikes.index(atm_strike)

        # Build OTM ladder: ATM, +1, +2, … up to otm_depth
        ladder: List[Dict] = []
        for offset in range(otm_depth + 1):          # 0 = ATM, 1..N = OTM
            idx = atm_idx + offset
            if idx >= len(sorted_strikes):
                break
            sk     = sorted_strikes[idx]
            ce_ltp = strike_map[sk]["CE_ltp"]
            ce_oi  = strike_map[sk]["CE_oi"]
            moneyness = "ATM" if offset == 0 else f"+{offset} OTM"
            ladder.append({
                "strike":    sk,
                "moneyness": moneyness,
                "CE_ltp":    round(ce_ltp, 2),
                "CE_oi":     ce_oi,
            })

        if not ladder:
            job_log(job_id, f"{ticker} no strikes found in ladder", "fail")
            continue

        # ── Premium Score ─────────────────────────────────────────────────────
        # Score = vol_surge_ratio × ATM_LTP.
        # At this point is_vol_surge is True (hard gate above), so vol_surge_ratio
        # is guaranteed to be >= min_vol_surge.  No fallback needed.
        # Higher score = more volume conviction at resistance + richer CE premium.
        atm_ltp       = ladder[0]["CE_ltp"]
        premium_score = round(vol_surge_ratio * atm_ltp, 2)

        # Best strike to SELL = first strike ABOVE resistance (existing logic)
        suggested_strike = sig.get("Suggested_Strike")

        job_log(job_id,
            f"{ticker} PREMIUM  score={premium_score:.1f}  "
            f"ATM Rs{atm_strike:.0f} CE={atm_ltp:.2f}  "
            f"vol={vol_label}  "
            f"sell_strike=Rs{suggested_strike or '?'}  expiry={expiry}",
            "signal")

        results.append({
            "Ticker":            ticker,
            "Current_Price":     round(cur_price, 2),
            "Surge_High":        round(resistance, 2),
            "Distance_Pct":      round(dist_pct, 2),
            "Vol_Surge_Ratio":        vol_surge_ratio,
            "Is_Vol_Surge":           is_vol_surge,
            "Today_Volume":           int(today_vol) if today_vol else None,
            "Avg_Vol_20d":            int(avg_vol_20d) if avg_vol_20d else None,
            "Today_High":             round(today_high, 2) if today_high else None,
            "High_Near_Resistance":   high_near_resistance,
            "High_Dist_Pct":          high_dist_pct,
            "Expiry":            expiry,
            "ATM_Strike":        ladder[0]["strike"]  if ladder else None,
            "ATM_CE_LTP":        ladder[0]["CE_ltp"]  if ladder else None,
            "ATM_CE_OI":         ladder[0]["CE_oi"]   if ladder else None,
            "OTM1_Strike":       ladder[1]["strike"]  if len(ladder) > 1 else None,
            "OTM1_CE_LTP":       ladder[1]["CE_ltp"]  if len(ladder) > 1 else None,
            "OTM1_CE_OI":        ladder[1]["CE_oi"]   if len(ladder) > 1 else None,
            "OTM2_Strike":       ladder[2]["strike"]  if len(ladder) > 2 else None,
            "OTM2_CE_LTP":       ladder[2]["CE_ltp"]  if len(ladder) > 2 else None,
            "OTM2_CE_OI":        ladder[2]["CE_oi"]   if len(ladder) > 2 else None,
            "OTM3_Strike":       ladder[3]["strike"]  if len(ladder) > 3 else None,
            "OTM3_CE_LTP":       ladder[3]["CE_ltp"]  if len(ladder) > 3 else None,
            "OTM3_CE_OI":        ladder[3]["CE_oi"]   if len(ladder) > 3 else None,
            "Sell_Strike":       suggested_strike,
            "Sell_CE_LTP":       sig.get("CE_LTP"),
            "Premium_Score":     premium_score,
            "Surge_Gain_Pct":    sig.get("Surge_Gain_Pct"),
            "G2_Date":           sig.get("Sell_Zone_EMA_Confirm_Date"),
            "Breakdown_Date":    sig.get("Breakdown_Date"),
            "Scanned_At":        now_ist_str(),
        })

        time.sleep(random.uniform(2.0, 4.0))

    # Sort by Premium Score descending (best sell opportunity first)
    results.sort(key=lambda x: x.get("Premium_Score", 0), reverse=True)

    job_log(job_id,
        f"Premium Zone done — {len(results)} qualified candidate(s)  "        f"({skipped_low_vol} skipped: vol below {min_vol_surge}x or HIGH not at resistance)",
        "info")
    job_progress(job_id, len(gate2), len(gate2), "")
    job_done(job_id, results)

    return results


# ─────────────────────────────────────────────────────────────
#  SCHEDULER
# ─────────────────────────────────────────────────────────────
_scheduler: Optional[BackgroundScheduler] = None

def start_scheduler(cfg: Dict):
    global _scheduler
    if _scheduler and _scheduler.running:
        try:
            _scheduler.shutdown(wait=False)
        except Exception:
            pass
    _scheduler = BackgroundScheduler(timezone=IST)

    def job():
        c = load_config()
        if c["market_hours_only"] and not is_market_open():
            return
        jid = create_job()
        run_screen_job(load_tickers(), c, jid)
        # Gate 2: check EMA+volume confirmation for all signals
        update_sell_zone_gates(c)
        all_sigs = load_signals()
        if all_sigs:
            hits = check_proximity_alerts(all_sigs, c)
            if hits:
                prox = load_proximity()
                # Deduplicate within 30-min window
                recent = {
                    h["Ticker"] for h in prox
                    if h.get("Alert_Time") and
                    (datetime.now(IST) -
                     datetime.strptime(h["Alert_Time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
                    ).total_seconds() < 1800
                }
                new_hits = [h for h in hits if h["Ticker"] not in recent]
                if new_hits:
                    prox.extend(new_hits)
                    save_proximity(prox[-200:])

    interval = max(cfg.get("auto_scan_interval_min", 15), 10)
    _scheduler.add_job(job, IntervalTrigger(minutes=interval),
                       id="main_scan", replace_existing=True)
    _scheduler.start()


# ─────────────────────────────────────────────────────────────
#  FASTAPI APP
# ─────────────────────────────────────────────────────────────
app = FastAPI(title="NSE Tools Suite", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/option-charts", include_in_schema=False)
async def option_charts_page():
    return FileResponse(BASE_DIR / "static" / "option-charts.html")

@app.get("/sma-screener", include_in_schema=False)
async def sma_screener_page():
    return FileResponse(BASE_DIR / "static" / "sma-screener.html")

@app.get("/backtest", include_in_schema=False)
async def backtest_page():
    return FileResponse(BASE_DIR / "static" / "backtest.html")

@app.get("/backtest-dashboard", include_in_schema=False)
async def backtest_dashboard_page():
    return FileResponse(BASE_DIR / "static" / "backtest_dashboard.html")

app.include_router(oc_router)
app.include_router(sma_router)

app.include_router(bt_router)   # ADD THIS


os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pydantic schemas ──────────────────────────────────────────
class ConfigUpdate(BaseModel):
    lookback_days:               Optional[int]   = None
    min_gain_percent:            Optional[float] = None
    min_green_candles:           Optional[int]   = None
    surge_recency_days:          Optional[int]   = None
    min_drop_percent:            Optional[float] = None
    min_breakdown_volume_ratio:  Optional[float] = None
    ema_period:                  Optional[int]   = None
    ema_filter_enabled:          Optional[bool]  = None
    price_proximity_percent:     Optional[float] = None
    ce_above_historical_high:    Optional[bool]  = None
    ce_history_days:             Optional[int]   = None
    max_signal_age_days:         Optional[int]   = None
    auto_scan_enabled:           Optional[bool]  = None
    auto_scan_interval_min:      Optional[int]   = None
    market_hours_only:           Optional[bool]  = None
    telegram_bot_token:          Optional[str]   = None
    telegram_chat_id:            Optional[str]   = None
    # Check 1 — Previous-Pullup-High filter
    prev_high_filter_enabled:        Optional[bool]  = None
    prev_high_lookback_days:         Optional[int]   = None
    prev_high_exclude_recent_days:   Optional[int]   = None
    prev_high_buffer_pct:            Optional[float] = None
    # Check 2 — Bear Call Spread
    bear_call_spread_width_intervals: Optional[int]  = None
    # Premium Sell Zone
    premium_min_vol_surge:            Optional[float] = None
    premium_otm_depth:                Optional[int]   = None

class SpecificScreenRequest(BaseModel):
    tickers: List[str]


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "static" / "index.html")

@app.get("/api/status")
async def get_status():
    cfg  = load_config()
    sigs = load_signals()
    prox = load_proximity()
    log  = load_scan_log()
    return {
        "market_open":      is_market_open(),
        "ist_time":         datetime.now(IST).strftime("%H:%M:%S"),
        "ist_date":         datetime.now(IST).strftime("%Y-%m-%d"),
        "active_signals":   len(sigs),
        "proximity_alerts": len(prox),
        "auto_scan":        cfg["auto_scan_enabled"],
        "scan_interval":    cfg["auto_scan_interval_min"],
        "last_scan":        log[-1]["time"] if log else None,
    }

@app.get("/api/config")
async def get_config():
    return load_config()

@app.put("/api/config")
async def update_config(data: ConfigUpdate):
    cfg     = load_config()
    updates = {k: v for k, v in data.dict().items() if v is not None}
    cfg.update(updates)
    save_config(cfg)
    if cfg["auto_scan_enabled"]:
        start_scheduler(cfg)
    return cfg

@app.get("/api/tickers")
async def get_tickers():
    return {"tickers": load_tickers()}

@app.post("/api/tickers/upload")
async def upload_tickers(file: UploadFile = File(...)):
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    if "SYMBOL" not in df.columns:
        raise HTTPException(400, "CSV must have a SYMBOL column")
    df.to_csv(TICKERS_FILE, index=False)
    tickers = [str(s).strip().upper() for s in df["SYMBOL"].dropna()]
    return {"count": len(tickers), "tickers": tickers[:20]}

@app.get("/api/signals")
async def get_signals():
    return load_signals()

@app.delete("/api/signals")
async def clear_signals():
    _wj(SIGNALS_FILE, [])
    _wj(PROXIMITY_FILE, [])
    return {"ok": True}

@app.get("/api/signals/gate-status")
async def get_gate_status():
    """Returns per-ticker gate status (Gates 1/2/3) for dashboard display."""
    sigs      = load_signals()
    cfg       = load_config()
    prox_pct  = cfg.get("price_proximity_percent", 8.0)
    result    = []
    for sig in sigs:
        ticker     = sig["Ticker"]
        resistance = (
            sig.get("Surge_High")
            or sig.get("Yesterday_High")
            or sig.get("Suggested_Strike")
        )
        cur_price  = None
        dist_pct   = None
        gate3      = False
        try:
            cur_price = get_current_price(ticker)
            if cur_price and resistance and cur_price <= resistance:
                dist_pct = round((resistance - cur_price) / resistance * 100, 2)
                gate3 = dist_pct <= prox_pct
        except Exception:
            pass
        result.append({
            "ticker":                    ticker,
            "gate1":                     True,  # Signal exists = Gate 1 passed
            "gate2":                     bool(sig.get("Sell_Zone_EMA_Confirmed")),
            "gate2_date":                sig.get("Sell_Zone_EMA_Confirm_Date"),
            "gate3":                     gate3,
            "current_price":             cur_price,
            "resistance":                resistance,
            "distance_pct":              dist_pct,
            "breakdown_date":            sig.get("Breakdown_Date"),
        })
    return result

@app.delete("/api/signals/{ticker}")
async def remove_signal(ticker: str):
    sigs = [s for s in load_signals() if s["Ticker"] != ticker.upper()]
    save_signals(sigs)
    return {"ok": True, "remaining": len(sigs)}


@app.post("/api/signals/{ticker}/ce-check")
async def run_ce_check(ticker: str):
    """
    On-demand CE historical-high check for a single signal.
    Called from the detail modal so the bulk scan doesn't pay this cost.
    """
    ticker = ticker.upper()
    sigs   = load_signals()
    sig    = next((s for s in sigs if s["Ticker"] == ticker), None)
    if not sig:
        raise HTTPException(404, f"Signal {ticker} not found")

    cfg    = load_config()
    if not cfg.get("ce_above_historical_high", True):
        return {"ticker": ticker, "remark": "check disabled", "ce_30d_high": None}

    strike = sig.get("Suggested_Strike")
    expiry = sig.get("Expiry")
    ce_ltp = sig.get("CE_LTP", 0)
    if not strike or not expiry:
        raise HTTPException(400, "Signal missing strike or expiry")

    ce_hist_high, _ = await asyncio.to_thread(
        get_ce_historical_high, ticker, strike, expiry, cfg.get("ce_history_days", 30)
    )

    if ce_hist_high is None:
        remark = "Unverified 30d High"
        status = "Unverified"
    elif ce_ltp <= ce_hist_high:
        remark = "CE Below 30d High"
        status = False
    else:
        remark = "CE at New High"
        status = True

    # Persist result back to signal
    for s in sigs:
        if s["Ticker"] == ticker:
            s["CE_30d_High"]       = round(ce_hist_high, 2) if ce_hist_high else "N/A"
            s["CE_Above_30d_High"] = status
            s["Remark"]            = remark
            s["CE_Hist_Checked"]   = True
            break
    save_signals(sigs)

    return {
        "ticker":       ticker,
        "ce_ltp":       ce_ltp,
        "ce_30d_high":  round(ce_hist_high, 2) if ce_hist_high else None,
        "remark":       remark,
        "status":       status,
    }

@app.get("/api/proximity")
async def get_proximity():
    return load_proximity()

@app.post("/api/proximity/check")
async def check_proximity():
    sigs = load_signals()
    if not sigs:
        return {"hits": 0, "alerts": [], "gate2_confirmed": 0}
    cfg  = load_config()
    # Gate 2: update EMA+volume confirmation before checking proximity
    gate2_new = await asyncio.to_thread(update_sell_zone_gates, cfg)
    sigs = load_signals()  # reload after gate2 update
    hits = await asyncio.to_thread(check_proximity_alerts, sigs, cfg)
    if hits:
        prox   = load_proximity()
        recent = {
            h["Ticker"] for h in prox
            if h.get("Alert_Time") and
            (datetime.now(IST) -
             datetime.strptime(h["Alert_Time"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=IST)
            ).total_seconds() < 1800
        }
        new_hits = [h for h in hits if h["Ticker"] not in recent]
        if new_hits:
            prox.extend(new_hits)
            save_proximity(prox[-200:])
        hits = new_hits
    return {"hits": len(hits), "alerts": hits, "gate2_confirmed": gate2_new}

@app.get("/api/scan-log")
async def get_scan_log():
    return list(reversed(load_scan_log()))

@app.post("/api/screen/all")
async def screen_all(background_tasks: BackgroundTasks):
    cfg     = load_config()
    tickers = load_tickers()
    job_id  = create_job()
    background_tasks.add_task(asyncio.to_thread, run_screen_job, tickers, cfg, job_id)
    return {"job_id": job_id, "total": len(tickers)}

@app.post("/api/screen/specific")
async def screen_specific(req: SpecificScreenRequest, background_tasks: BackgroundTasks):
    cfg     = load_config()
    tickers = [t.strip().upper() for t in req.tickers if t.strip()]
    job_id  = create_job()
    background_tasks.add_task(asyncio.to_thread, run_screen_job, tickers, cfg, job_id)
    return {"job_id": job_id, "total": len(tickers)}

@app.get("/api/jobs/{job_id}/stream")
async def job_stream(job_id: str):
    async def event_gen():
        sent = 0
        while True:
            with jobs_lock:
                job = jobs.get(job_id)
            if job is None:
                yield f"data: {json.dumps({'type':'error','msg':'Job not found'})}\n\n"
                break
            logs = job.get("logs", [])
            for entry in logs[sent:]:
                yield f"data: {json.dumps({'type':'log','time':entry['time'],'msg':entry['msg'],'level':entry['level']})}\n\n"
            sent = len(logs)
            yield f"data: {json.dumps({'type':'progress','current':job['progress'],'total':job['total'],'ticker':job.get('current_ticker','')})}\n\n"
            if job["status"] in ("done", "error"):
                yield f"data: {json.dumps({'type':'done','status':job['status'],'count':len(job.get('result',[]))})}\n\n"
                break
            await asyncio.sleep(0.6)

    return StreamingResponse(event_gen(), media_type="text/event-stream",
                              headers={"Cache-Control": "no-cache",
                                       "X-Accel-Buffering": "no"})

@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job

@app.get("/api/chart/{ticker}")
async def get_chart_data(ticker: str, days: int = 60):
    hist = await asyncio.to_thread(get_price_history, ticker.upper(), days)
    if hist is None or hist.empty:
        raise HTTPException(404, f"No data for {ticker}")

    signals = load_signals()
    sig = next((s for s in signals if s["Ticker"] == ticker.upper()), None)

    data = []
    for idx, row in hist.iterrows():
        data.append({
            "date":   idx.strftime("%Y-%m-%d"),
            "open":   round(float(row["Open"]),   2),
            "high":   round(float(row["High"]),   2),
            "low":    round(float(row["Low"]),    2),
            "close":  round(float(row["Close"]),  2),
            "volume": int(row.get("Volume", 0)),
        })

    return {
        "ticker":          ticker.upper(),
        "strike":          sig.get("Suggested_Strike")  if sig else None,
        "surge_high":      sig.get("Surge_High")        if sig else None,  # TRUE resistance (surge peak)
        "yesterday_high":  sig.get("Yesterday_High")    if sig else None,  # Day before breakdown
        "surge_start":     sig.get("Surge_Start")       if sig else None,  # for surge highlight
        "surge_end":       sig.get("Surge_End")         if sig else None,
        "candles":         data,
    }


# ─────────────────────────────────────────────────────────────
#  PREMIUM SELL ZONE ROUTES
# ─────────────────────────────────────────────────────────────
@app.get("/api/premium-zone")
async def get_premium_zone_data():
    """Return the last saved Premium Zone scan results."""
    return load_premium_zone()


@app.post("/api/premium-zone/scan")
async def start_premium_zone_scan(background_tasks: BackgroundTasks):
    """
    Launch a background Premium Zone scan.
    Returns a job_id for SSE streaming (reuses existing /api/jobs/{id}/stream).
    Results are saved to premium_zone.json when the job completes.
    """
    cfg    = load_config()
    job_id = create_job()

    async def _run():
        results = await asyncio.to_thread(scan_premium_zone_job, cfg, job_id)
        save_premium_zone(results)

    background_tasks.add_task(_run)
    return {"job_id": job_id}


@app.delete("/api/premium-zone")
async def clear_premium_zone():
    """Clear saved Premium Zone results."""
    save_premium_zone([])
    return {"ok": True}


# ─────────────────────────────────────────────────────────────
#  STARTUP
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    cfg = load_config()
    if cfg.get("auto_scan_enabled"):
        start_scheduler(cfg)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)