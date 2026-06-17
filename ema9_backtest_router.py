"""
9EMA Breakout Backtesting Router — FastAPI
══════════════════════════════════════════
Strategy Rules (exact match of ema9_router.py signal logic):
  1. Previous candle close  < 9 EMA          → was below
  2. Breakout candle close  > 9 EMA          → crossed above
  3. Confirmation candle close > breakout close → momentum confirmed
  4. Entry price (open of candle after confirmation) < composite_fair_price
     → only trade UNDERVALUED / FAIR stocks, never overvalued
  5. TARGET: +N% from entry (default 3%)
  6. STOP:   Candle closes below 9 EMA

Routes:
  POST /api/backtest/ema9/run   → full backtest for N tickers
  GET  /api/backtest/ema9/stats → aggregate stats from last run (in-memory)
"""

import os, asyncio, datetime as dt_module, math
from typing import Optional, List, Dict, Any

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# ── Fair Value import from sma_router ──────────────────────────
try:
    from sma_router import _analyze_ticker as _sma_analyze_ticker
    _FV_AVAILABLE = True
except ImportError:
    _FV_AVAILABLE = False

router = APIRouter()

# ─────────────────────────────────────────────────────────────
#  RATE LIMIT CONSTANTS  (same as ema9_router.py)
# ─────────────────────────────────────────────────────────────
YF_CHUNK_SIZE  = 100
YF_CHUNK_DELAY = 2.0
FV_INTER_DELAY = 1.5

# ─────────────────────────────────────────────────────────────
#  EMA CALCULATION
#  Mirrors pandas ewm(span=9, adjust=False) — alpha = 2/(span+1)
# ─────────────────────────────────────────────────────────────
def _calc_ema9(closes: List[float]) -> List[float]:
    alpha = 2 / (9 + 1)
    ema   = [0.0] * len(closes)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]
    return ema


# ─────────────────────────────────────────────────────────────
#  BATCH PRICE DOWNLOAD  (reuses ema9_router pattern)
# ─────────────────────────────────────────────────────────────
def _batch_download(
    tickers: List[str],
    interval: str,
    lookback_days: int,
) -> Dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    yf_symbols = [f"{t}.NS" for t in tickers]
    end   = dt_module.date.today()
    start = end - dt_module.timedelta(days=lookback_days + 30)
    try:
        raw = yf.download(
            yf_symbols,
            start=str(start), end=str(end),
            interval=interval,
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=False,
        )
    except Exception:
        return {}
    if raw is None or raw.empty:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for ticker, sym in zip(tickers, yf_symbols):
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                df = raw[sym][["Open", "High", "Low", "Close", "Volume"]].copy()
            else:
                df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.dropna(how="all")
            if len(df) >= 15:
                result[ticker] = df
        except Exception:
            continue
    return result


# ─────────────────────────────────────────────────────────────
#  FAIR VALUE  (same as ema9_router._enrich_fair_value)
# ─────────────────────────────────────────────────────────────
def _fetch_fair_value(ticker: str) -> Optional[float]:
    """
    Returns composite_fair_price (float) or None if unavailable.
    """
    if not _FV_AVAILABLE:
        return None
    try:
        res = _sma_analyze_ticker(ticker, fy_start=2014, force=False, include_other_income=True)
        if "error" in res:
            return None
        fv = res.get("composite_fair_price")
        return float(fv) if fv is not None else None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────
#  CORE BACKTESTING ENGINE
# ─────────────────────────────────────────────────────────────
def _backtest_ticker(
    ticker:         str,
    df:             pd.DataFrame,
    fair_price:     Optional[float],
    target_pct:     float = 3.0,
    max_hold:       int   = 20,
) -> List[Dict[str, Any]]:
    """
    Walk-forward backtest for a single ticker.
    Returns a list of trade dicts (one per signal found).

    Trade outcome labels:
      SKIPPED  — entry price >= fair_price (overvalued, no trade taken)
      WIN      — target hit (high of exit candle >= target_price)
      LOSS     — stop hit (candle closes below 9 EMA)
      OPEN     — max_hold expired (exited at last candle's close, mark-to-market)
    """
    df = df.copy().dropna()
    if len(df) < 15:
        return []

    closes = df["Close"].tolist()
    opens  = df["Open"].tolist()
    highs  = df["High"].tolist()
    lows   = df["Low"].tolist()
    n      = len(closes)

    def _date(i: int) -> str:
        d = df.index[i]
        return str(d.date()) if hasattr(d, "date") else str(d)[:10]

    ema9 = _calc_ema9(closes)
    target_mult = 1 + target_pct / 100.0

    trades: List[Dict[str, Any]] = []
    in_trade_until = -1

    for i in range(1, n - 2):
        # ── Skip if already in a trade ────────────────────────────────
        if i <= in_trade_until:
            continue

        prev_close = closes[i - 1]
        prev_ema   = ema9[i - 1]
        curr_close = closes[i]
        curr_ema   = ema9[i]
        conf_close = closes[i + 1]

        # ── SIGNAL CONDITION ──────────────────────────────────────────
        # (exact match of ema9_router._process_ticker_df)
        is_breakout = (
            prev_close < prev_ema     # previous candle below EMA
            and curr_close > curr_ema  # breakout candle above EMA
            and conf_close > curr_close # confirmation candle above breakout
        )
        if not is_breakout:
            continue

        # Entry is OPEN of candle AFTER confirmation (i+2)
        if i + 2 >= n:
            continue

        entry_price = opens[i + 2]
        entry_date  = _date(i + 2)
        entry_ema9  = ema9[i + 2]

        # ── FAIR VALUE GATE ───────────────────────────────────────────
        has_fv     = fair_price is not None and fair_price > 0
        below_fv   = (entry_price < fair_price) if has_fv else True
        fair_gap   = round((fair_price - entry_price) / entry_price * 100, 2) if has_fv else None

        if has_fv and not below_fv:
            # Record SKIPPED trade — overvalued entry
            trades.append({
                "ticker":         ticker,
                "outcome":        "SKIPPED",
                "stop_reason":    "OVERVALUED",
                "breakout_date":  _date(i),
                "breakout_close": round(curr_close, 2),
                "confirm_date":   _date(i + 1),
                "confirm_close":  round(conf_close, 2),
                "entry_date":     entry_date,
                "entry_price":    round(entry_price, 2),
                "entry_ema9":     round(entry_ema9, 2),
                "fair_price":     round(fair_price, 2) if has_fv else None,
                "fair_gap_pct":   fair_gap,
                "target_price":   round(entry_price * target_mult, 2),
                "exit_date":      None,
                "exit_price":     None,
                "pnl_pct":        None,
                "hold_candles":   None,
            })
            continue  # no trade taken

        # ── TRADE SIMULATION ─────────────────────────────────────────
        target_price = entry_price * target_mult
        exit_price   = None
        exit_date    = None
        outcome      = "OPEN"
        stop_reason  = None
        exit_idx     = None

        for j in range(i + 3, min(n, i + 3 + max_hold)):
            h = highs[j]
            c = closes[j]
            e = ema9[j]

            # TARGET HIT: any intraday high touches or exceeds target
            if h >= target_price:
                exit_price  = round(target_price, 2)
                exit_date   = _date(j)
                outcome     = "WIN"
                stop_reason = "TARGET"
                exit_idx    = j
                in_trade_until = j
                break

            # STOP: candle closes below 9 EMA — trend invalidated
            if c < e:
                exit_price  = round(c, 2)
                exit_date   = _date(j)
                outcome     = "LOSS"
                stop_reason = "EMA_BREAK"
                exit_idx    = j
                in_trade_until = j
                break

        # Max hold expiry
        if outcome == "OPEN":
            last_j = min(i + 2 + max_hold, n - 1)
            if last_j > i + 2:
                exit_price     = round(closes[last_j], 2)
                exit_date      = _date(last_j)
                exit_idx       = last_j
                outcome        = "WIN" if exit_price >= entry_price else "LOSS"
                stop_reason    = "MAX_HOLD"
                in_trade_until = last_j

        pnl_pct     = round((exit_price - entry_price) / entry_price * 100, 2) if exit_price else None
        hold_candles= (exit_idx - (i + 2)) if exit_idx is not None else None

        trades.append({
            "ticker":         ticker,
            "outcome":        outcome,
            "stop_reason":    stop_reason,
            "breakout_date":  _date(i),
            "breakout_close": round(curr_close, 2),
            "confirm_date":   _date(i + 1),
            "confirm_close":  round(conf_close, 2),
            "entry_date":     entry_date,
            "entry_price":    round(entry_price, 2),
            "entry_ema9":     round(entry_ema9, 2),
            "fair_price":     round(fair_price, 2) if has_fv else None,
            "fair_gap_pct":   fair_gap,
            "target_price":   round(target_price, 2),
            "exit_date":      exit_date,
            "exit_price":     exit_price,
            "pnl_pct":        pnl_pct,
            "hold_candles":   hold_candles,
        })

        # Skip forward past exit to avoid overlapping trades
        if exit_idx is not None:
            i = exit_idx  # loop will i++ on next iteration

    return trades


# ─────────────────────────────────────────────────────────────
#  AGGREGATE STATS COMPUTATION
# ─────────────────────────────────────────────────────────────
def _compute_stats(trades: List[Dict]) -> Dict:
    """
    Compute aggregate statistics for a list of trade dicts.
    """
    taken   = [t for t in trades if t["outcome"] != "SKIPPED"]
    wins    = [t for t in taken  if t["outcome"] == "WIN"]
    losses  = [t for t in taken  if t["outcome"] == "LOSS"]
    open_t  = [t for t in taken  if t["outcome"] == "OPEN"]
    skipped = [t for t in trades if t["outcome"] == "SKIPPED"]
    closed  = wins + losses

    win_rate  = (len(wins) / len(closed) * 100) if closed else 0.0
    avg_win   = (sum(t["pnl_pct"] for t in wins   if t["pnl_pct"] is not None) / len(wins))   if wins   else 0.0
    avg_loss  = (sum(t["pnl_pct"] for t in losses if t["pnl_pct"] is not None) / len(losses)) if losses else 0.0
    total_pnl = sum(t["pnl_pct"] for t in closed  if t["pnl_pct"] is not None)
    avg_hold  = (sum(t["hold_candles"] for t in taken if t["hold_candles"] is not None) / max(1, len([t for t in taken if t["hold_candles"] is not None])))

    profit_factor = None
    if avg_loss != 0 and wins:
        profit_factor = round(abs(avg_win / avg_loss), 3)

    # Per-ticker breakdown
    ticker_map: Dict[str, List] = {}
    for t in trades:
        ticker_map.setdefault(t["ticker"], []).append(t)

    per_ticker = []
    for ticker, tt in sorted(ticker_map.items()):
        ts = _compute_stats.__wrapped__(tt) if hasattr(_compute_stats, "__wrapped__") else None
        t_taken   = [x for x in tt if x["outcome"] != "SKIPPED"]
        t_wins    = [x for x in t_taken if x["outcome"] == "WIN"]
        t_losses  = [x for x in t_taken if x["outcome"] == "LOSS"]
        t_closed  = t_wins + t_losses
        per_ticker.append({
            "ticker":      ticker,
            "total":       len(tt),
            "taken":       len(t_taken),
            "wins":        len(t_wins),
            "losses":      len(t_losses),
            "open":        len([x for x in t_taken if x["outcome"] == "OPEN"]),
            "skipped":     len([x for x in tt if x["outcome"] == "SKIPPED"]),
            "win_rate":    round(len(t_wins) / len(t_closed) * 100, 1) if t_closed else 0,
            "total_pnl":   round(sum(x["pnl_pct"] for x in t_closed if x["pnl_pct"] is not None), 2),
        })

    return {
        "total_signals":  len(trades),
        "trades_taken":   len(taken),
        "wins":           len(wins),
        "losses":         len(losses),
        "open":           len(open_t),
        "skipped":        len(skipped),
        "win_rate":       round(win_rate, 2),
        "avg_win_pct":    round(avg_win, 2),
        "avg_loss_pct":   round(avg_loss, 2),
        "total_pnl_pct":  round(total_pnl, 2),
        "avg_hold_candles": round(avg_hold, 1),
        "profit_factor":  profit_factor,
        "per_ticker":     per_ticker,
    }


# ─────────────────────────────────────────────────────────────
#  REQUEST / RESPONSE MODELS
# ─────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    tickers:       List[str]
    interval:      str  = "1d"       # "1d" | "1wk"
    lookback_days: int  = 365        # how far back to scan
    target_pct:    float = 3.0       # profit target %
    max_hold:      int  = 20         # max candles before force-exit
    require_fv:    bool = True       # if True, skip trades without FV data


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@router.post("/api/backtest/ema9/run")
async def backtest_run(req: BacktestRequest):
    """
    Full walk-forward backtest for one or more tickers.

    Returns:
      - trades:    list of all trade records
      - stats:     aggregate statistics
      - per_ticker: per-ticker breakdown
      - failed:    tickers with data errors
    """
    tickers = [t.strip().upper() for t in req.tickers if t.strip()][:200]
    if not tickers:
        raise HTTPException(400, "No tickers provided.")

    all_trades: List[Dict]   = []
    failed:     List[Dict]   = []
    n_chunks = math.ceil(len(tickers) / YF_CHUNK_SIZE)

    # ── Step 1: Chunked batch price download ─────────────────────────
    ticker_dfs: Dict[str, pd.DataFrame] = {}
    for ci in range(n_chunks):
        chunk = tickers[ci * YF_CHUNK_SIZE : (ci + 1) * YF_CHUNK_SIZE]
        chunk_dfs = await asyncio.to_thread(
            _batch_download, chunk, req.interval, req.lookback_days
        )
        ticker_dfs.update(chunk_dfs)
        if ci < n_chunks - 1:
            await asyncio.sleep(YF_CHUNK_DELAY)

    # ── Step 2: Fetch fair value (serial, rate-limited) ───────────────
    fv_map: Dict[str, Optional[float]] = {}
    for i, ticker in enumerate(tickers):
        if ticker not in ticker_dfs:
            continue
        fv = await asyncio.to_thread(_fetch_fair_value, ticker)
        fv_map[ticker] = fv
        if i < len(tickers) - 1:
            await asyncio.sleep(FV_INTER_DELAY)

    # ── Step 3: Run backtest per ticker ──────────────────────────────
    for ticker in tickers:
        if ticker not in ticker_dfs:
            failed.append({"ticker": ticker, "error": "No price data"})
            continue
        try:
            fv = fv_map.get(ticker)
            # If require_fv is True and no FV available, still run but mark trades
            trades = _backtest_ticker(
                ticker       = ticker,
                df           = ticker_dfs[ticker],
                fair_price   = fv,
                target_pct   = req.target_pct,
                max_hold     = req.max_hold,
            )
            all_trades.extend(trades)
        except Exception as exc:
            failed.append({"ticker": ticker, "error": str(exc)})

    # ── Step 4: Stats aggregation ────────────────────────────────────
    stats = _compute_stats(all_trades)

    # Sort trades: most recent entry first
    all_trades.sort(key=lambda t: t.get("entry_date") or "", reverse=True)

    return {
        "trades":     all_trades,
        "stats":      stats,
        "per_ticker": stats.pop("per_ticker", []),
        "failed":     failed,
        "config": {
            "tickers":       tickers,
            "interval":      req.interval,
            "lookback_days": req.lookback_days,
            "target_pct":    req.target_pct,
            "max_hold":      req.max_hold,
        },
    }


@router.get("/api/backtest/ema9/strategy")
async def backtest_strategy_info():
    """
    Returns a human-readable description of the backtesting strategy.
    """
    return {
        "name":    "9EMA Breakout with Fair Value Gate",
        "version": "1.0",
        "rules": [
            "1. Previous candle close BELOW 9 EMA",
            "2. Breakout candle close ABOVE 9 EMA",
            "3. Confirmation candle close ABOVE breakout candle close",
            "4. Entry price (next open) BELOW composite fair value (Screener.in regression)",
            "5. Target: +N% from entry (default 3%)",
            "6. Stop: Candle closes below 9 EMA",
            "7. Max hold: N candles then force-exit at close",
        ],
        "entry":  "Open of candle after confirmation (i+2)",
        "target": "3% above entry (configurable)",
        "stop":   "Candle close below 9 EMA",
        "fv_gate":"Entry price must be below composite_fair_price from sma_router",
        "overlap": "Non-overlapping trades only (next signal starts after current exit)",
    }
