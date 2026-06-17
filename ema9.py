"""
9 EMA Breakout Screener v3 — FastAPI Router (Production-Ready)
═══════════════════════════════════════════════════════════════
Signal logic (ALL four must be true):
  1. Long-Term Trend: Current Close > 50-day SMA
  2. 9 EMA Breakout: Candle n-1 closes ABOVE 9 EMA, previous candle was BELOW 9 EMA
  3. Confirmation: Candle n closes ABOVE breakout candle's close (Higher High)
  4. 3% Target: Auto-calculated as Current Price * 1.03

Fair Value Overlay:
  - Prime Target:  Signal == True AND Price < Fair Value
  - Technical Only: Signal == True AND Price > Fair Value

Routes:
  POST /api/ema9/upload-csv  → Upload CSV, return extracted symbols
  GET  /api/ema9/tickers       → autocomplete
  GET  /api/ema9/tickers/list  → full list
  POST /api/ema9/screen        → batch screener (with trend + FV + 3% target)
"""

import io, os, asyncio, datetime as dt_module, math
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

try:
    from sma_router import _analyze_ticker as _sma_analyze_ticker
    _FV_AVAILABLE = True
except ImportError:
    _FV_AVAILABLE = False

router = APIRouter()

DATA_DIR = "data"
FNO_CSV  = "tickers.csv"
ALL_CSV  = "tickers_all.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  RATE LIMIT CONFIG
# ─────────────────────────────────────────────────────────────
YF_CHUNK_SIZE    = 100
YF_CHUNK_DELAY   = 2.0
FV_INTER_DELAY   = 1.5
TREND_LOOKBACK_DAYS = 400

# ─────────────────────────────────────────────────────────────
#  TICKER LOADING
# ─────────────────────────────────────────────────────────────
_fno_df: Optional[pd.DataFrame] = None
_all_df: Optional[pd.DataFrame] = None


def _load_csv_df(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        return pd.DataFrame(columns=["symbol", "company_name"])
    try:
        df = pd.read_csv(csv_path, dtype=str)
        df.columns = [c.strip().lower() for c in df.columns]
        if "name of company" in df.columns:
            df = df.rename(columns={"name of company": "company_name"})
        elif "security" in df.columns:
            df = df.rename(columns={"security": "company_name"})
        elif "company_name" not in df.columns:
            df["company_name"] = ""
        if "symbol" not in df.columns:
            return pd.DataFrame(columns=["symbol", "company_name"])
        df["symbol"]       = df["symbol"].str.strip().str.upper()
        df["company_name"] = df.get("company_name", pd.Series([""] * len(df))).fillna("").str.strip()
        out = df[["symbol", "company_name"]].dropna(subset=["symbol"])
        return out[out["symbol"] != ""].reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=["symbol", "company_name"])


def _load_fno_df() -> pd.DataFrame:
    global _fno_df
    if _fno_df is None:
        _fno_df = _load_csv_df(FNO_CSV)
    return _fno_df


def _load_all_df() -> pd.DataFrame:
    global _all_df
    if _all_df is None:
        _all_df = _load_csv_df(ALL_CSV)
    return _all_df


# ─────────────────────────────────────────────────────────────
#  CSV UPLOAD — Robust parsing with BOM handling
# ─────────────────────────────────────────────────────────────
@router.post("/api/ema9/upload-csv")
async def ema9_upload_csv(file: UploadFile = File(...)):
    """
    Accept a multipart CSV upload. Robust parsing:
      - encoding='utf-8-sig' to strip BOM from third-party CSVs
      - Dynamic column mapping: finds Symbol / Ticker / Stock (case-insensitive)
    Returns: {symbols: ["RELIANCE", "TCS", ...], count: N, source: "upload"}
    """
    try:
        content = await file.read()
        # Robust: utf-8-sig strips BOM; thousands separator comma handled
        df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig", thousands=",")
    except Exception as exc:
        raise HTTPException(400, f"CSV parse error: {exc}")

    # Dynamic column mapping — find Symbol/Ticker/Stock column
    cols_lower = {c.strip().lower(): c for c in df.columns}
    symbol_col = None
    for key in ("symbol", "ticker", "stock"):
        if key in cols_lower:
            symbol_col = cols_lower[key]
            break

    if symbol_col is None:
        raise HTTPException(400, "Could not find Symbol/Ticker/Stock column in CSV.")

    symbols = (
        df[symbol_col]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(r"\.NS$", "", regex=True)
        .replace(["NAN", "NONE", ""], pd.NA)
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "symbols": symbols,
        "count": len(symbols),
        "source": "upload",
        "filename": file.filename,
    }


# ─────────────────────────────────────────────────────────────
#  TREND DETECTION
# ─────────────────────────────────────────────────────────────
def _compute_trend(df: pd.DataFrame) -> Dict:
    """
    Compute trend regime from OHLCV DataFrame (needs >= 50 candles for 50SMA).
    Returns: trend_regime, sma50, sma200, lr_slope_pct, above_50sma, above_200sma
    """
    n = len(df)
    close = df["Close"]

    sma50  = float(close.rolling(50).mean().iloc[-1])  if n >= 50  else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else None
    price  = float(close.iloc[-1])

    # Linear regression slope on last 60 candles (% per candle)
    lr_slope_pct = 0.0
    if n >= 30:
        window = min(60, n)
        y  = close.iloc[-window:].values
        x  = np.arange(window, dtype=float)
        if y.mean() != 0:
            slope = np.polyfit(x, y, 1)[0]
            lr_slope_pct = round(slope / y.mean() * 100, 4)

    above_50sma  = (price > sma50)  if sma50  is not None else None
    above_200sma = (price > sma200) if sma200 is not None else None

    # Determine regime
    if sma50 is not None and sma200 is not None:
        if above_50sma and sma50 > sma200 and lr_slope_pct > -0.05:
            regime = "UPTREND"
        elif not above_50sma and sma50 < sma200 and lr_slope_pct < 0.05:
            regime = "DOWNTREND"
        else:
            regime = "SIDEWAYS"
    elif sma50 is not None:
        if above_50sma and lr_slope_pct > 0:
            regime = "UPTREND"
        elif not above_50sma and lr_slope_pct < 0:
            regime = "DOWNTREND"
        else:
            regime = "SIDEWAYS"
    else:
        regime = "UPTREND" if lr_slope_pct > 0.05 else ("DOWNTREND" if lr_slope_pct < -0.05 else "SIDEWAYS")

    return {
        "trend_regime":   regime,
        "sma50":          round(sma50,  2) if sma50  is not None else None,
        "sma200":         round(sma200, 2) if sma200 is not None else None,
        "lr_slope_pct":   lr_slope_pct,
        "above_50sma":    above_50sma,
        "above_200sma":   above_200sma,
    }


# ─────────────────────────────────────────────────────────────
#  EMA9 BREAKOUT CORE LOGIC
# ─────────────────────────────────────────────────────────────
def _process_ticker_df(ticker: str, df: pd.DataFrame, max_candles_ago: int) -> Dict:
    """
    Run EMA9 breakout + trend detection on a pre-downloaded OHLCV DataFrame.
    Requires columns: Open, High, Low, Close, Volume (DatetimeIndex).
    """
    df = df.copy().dropna()
    if len(df) < 60:
        return {"ticker": ticker, "status": "NO_DATA", "error": "Not enough candles (<60 days)"}

    # Compute indicators
    df["ema9"]  = df["Close"].ewm(span=9,   adjust=False).mean()
    df["sma50"] = df["Close"].rolling(50).mean()

    n             = len(df)
    current_price = round(float(df["Close"].iloc[-1]), 2)
    current_ema9  = round(float(df["ema9"].iloc[-1]),  2)

    # ── Trend detection ──────────────────────────────────────
    trend_data = _compute_trend(df)

    # ── Long-Term Trend Filter: Close > 50 SMA ─────────────────
    if trend_data["sma50"] is not None and current_price <= trend_data["sma50"]:
        return {
            "ticker":       ticker,
            "status":       "NO_SIGNAL",
            "reason":       "Price below 50 SMA",
            "current_price": current_price,
            "ema9":          current_ema9,
            **trend_data,
        }

    # ── EMA9 breakout scan ───────────────────────────────────
    # Look for: candle i-1 closes below 9 EMA, candle i closes above 9 EMA,
    #           candle i+1 closes above candle i's close (Higher High)
    scan_start = max(1, n - max_candles_ago - 2)
    scan_end   = n - 2

    found = False
    breakout_idx = confirm_idx = None

    for i in range(scan_end, scan_start - 1, -1):
        # Candle i-1: previous candle (below 9 EMA)
        prev_close = float(df["Close"].iloc[i - 1])
        prev_ema   = float(df["ema9"].iloc[i - 1])
        # Candle i: breakout candle (above 9 EMA)
        curr_close = float(df["Close"].iloc[i])
        curr_ema   = float(df["ema9"].iloc[i])
        # Candle i+1: confirmation candle (higher close than breakout)
        conf_close = float(df["Close"].iloc[i + 1])

        # Explicit "Higher High" confirmation: conf_close > curr_close
        if prev_close < prev_ema and curr_close > curr_ema and conf_close > curr_close:
            found = True
            breakout_idx = i
            confirm_idx  = i + 1
            break

    if not found:
        return {
            "ticker":       ticker,
            "status":       "NO_SIGNAL",
            "current_price": current_price,
            "ema9":          current_ema9,
            **trend_data,
        }

    breakout_candle = df.iloc[breakout_idx]
    confirm_candle  = df.iloc[confirm_idx]

    def _date(idx):
        d = df.index[idx]
        return str(d.date()) if hasattr(d, "date") else str(d)[:10]

    bo_date  = _date(breakout_idx)
    con_date = _date(confirm_idx)

    # ── 3% Target calculation ──────────────────────────────────
    target_3pct = round(current_price * 1.03, 2)
    gap_to_target_pct = round((target_3pct - current_price) / current_price * 100, 2)
    target_nearing = abs(gap_to_target_pct - 3.0) <= 0.5

    # ── Last 60 candles for chart ─────────────────────────────
    chart_start = max(0, n - 60)
    candles = []
    for j in range(chart_start, n):
        row = df.iloc[j]
        candles.append({
            "date":        str(df.index[j].date()) if hasattr(df.index[j], "date") else str(df.index[j])[:10],
            "open":        round(float(row["Open"]),  2),
            "high":        round(float(row["High"]),  2),
            "low":         round(float(row["Low"]),   2),
            "close":       round(float(row["Close"]), 2),
            "volume":      int(row["Volume"]),
            "ema9":        round(float(df["ema9"].iloc[j]),  2),
            "sma50":       round(float(df["sma50"].iloc[j]), 2) if not pd.isna(df["sma50"].iloc[j]) else None,
            "is_breakout": (_date(j) == bo_date),
            "is_confirm":  (_date(j) == con_date),
        })

    return {
        "ticker":         ticker,
        "status":         "SIGNAL",
        "current_price":  current_price,
        "ema9":           current_ema9,
        "ema9_dist_pct":  round(abs(current_price - current_ema9) / current_ema9 * 100, 2),
        "target_3pct":    target_3pct,
        "target_nearing": target_nearing,
        "breakout_date":  bo_date,
        "breakout_close": round(float(breakout_candle["Close"]), 2),
        "breakout_high":  round(float(breakout_candle["High"]),  2),
        "confirm_date":   con_date,
        "confirm_close":  round(float(confirm_candle["Close"]), 2),
        "candles_ago":    n - 1 - confirm_idx,
        "interval":       "batch",
        "candles":        candles,
        **trend_data,
    }


# ─────────────────────────────────────────────────────────────
#  BATCH yf.DOWNLOAD
# ─────────────────────────────────────────────────────────────
def _batch_download(
    tickers: List[str],
    interval: str,
    lookback_days: int,
) -> Dict[str, pd.DataFrame]:
    """
    Single yf.download call for up to YF_CHUNK_SIZE tickers.
    Returns {ticker: ohlcv_df}.
    """
    if not tickers:
        return {}

    effective_days = max(lookback_days, TREND_LOOKBACK_DAYS)
    yf_symbols = [f"{t}.NS" for t in tickers]
    end   = dt_module.date.today()
    start = end - dt_module.timedelta(days=effective_days + 30)

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
            if len(df) >= 60:
                result[ticker] = df
        except Exception:
            continue
    return result


# ─────────────────────────────────────────────────────────────
#  SINGLE-TICKER FALLBACK
# ─────────────────────────────────────────────────────────────
def _screen_ticker(
    ticker: str,
    interval: str = "1d",
    lookback_days: int = 180,
    max_candles_ago: int = 10,
) -> Dict:
    ticker    = ticker.strip().upper()
    yf_symbol = f"{ticker}.NS"
    effective_days = max(lookback_days, TREND_LOOKBACK_DAYS)
    try:
        end   = dt_module.date.today()
        start = end - dt_module.timedelta(days=effective_days + 30)
        raw   = yf.download(
            yf_symbol, start=str(start), end=str(end),
            interval=interval, progress=False, auto_adjust=True, threads=False,
        )
    except Exception as exc:
        return {"ticker": ticker, "status": "ERROR", "error": str(exc)}

    if raw is None or raw.empty or len(raw) < 60:
        return {"ticker": ticker, "status": "NO_DATA", "error": "Insufficient price data (<60 days)"}

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
    return _process_ticker_df(ticker, df, max_candles_ago)


# ─────────────────────────────────────────────────────────────
#  FAIR VALUE ENRICHMENT
# ─────────────────────────────────────────────────────────────
_FV_NULL = {
    "composite_fair_price": None,
    "composite_gain_pct":   None,
    "fair_gap_pct":         None,
    "valuation_bucket":     "N/A",
    "fv_model_count":       0,
}


def _enrich_fair_value(ticker: str) -> Dict:
    if not _FV_AVAILABLE:
        return _FV_NULL
    try:
        res = _sma_analyze_ticker(
            ticker,
            fy_start=2014,
            force=False,
            include_other_income=True,
        )
        if "error" in res:
            return _FV_NULL
        comp_fair = res.get("composite_fair_price")
        comp_gain = res.get("composite_gain_pct")
        bucket    = res.get("valuation_bucket", "N/A")
        model_cnt = res.get("model_count", 0)
        if comp_fair is None or comp_gain is None:
            return _FV_NULL
        return {
            "composite_fair_price": comp_fair,
            "composite_gain_pct":   comp_gain,
            "fair_gap_pct":         comp_gain,
            "valuation_bucket":     bucket,
            "fv_model_count":       model_cnt,
        }
    except Exception:
        return _FV_NULL


# ─────────────────────────────────────────────────────────────
#  REQUEST MODEL
# ─────────────────────────────────────────────────────────────
class Ema9ScreenRequest(BaseModel):
    tickers:               List[str]
    interval:              str  = "1d"
    lookback_days:         int  = 180
    max_candles_ago:       int  = 10
    require_uptrend:       bool = True
    allow_sideways:        bool = False


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@router.get("/api/ema9/tickers")
async def ema9_tickers(q: str = "", source: str = "fno"):
    df = _load_all_df() if source == "all" else _load_fno_df()
    if df.empty:
        return {"tickers": [], "total": 0}
    q = q.strip().upper()
    if q:
        mask = df["symbol"].str.contains(q, na=False)
        if "company_name" in df.columns:
            mask = mask | df["company_name"].str.upper().str.contains(q, na=False)
        filtered = df[mask].head(30)
    else:
        filtered = df.head(30)
    return {
        "total": len(df),
        "tickers": [
            {"symbol": row["symbol"], "name": row["company_name"] or row["symbol"]}
            for _, row in filtered.iterrows()
        ],
    }


@router.get("/api/ema9/tickers/list")
async def ema9_tickers_list(source: str = "fno"):
    df = _load_all_df() if source == "all" else _load_fno_df()
    return {
        "source":  source,
        "total":   len(df),
        "symbols": df["symbol"].tolist(),
        "tickers": [
            {"symbol": row["symbol"], "name": row["company_name"] or row["symbol"]}
            for _, row in df.iterrows()
        ],
    }


@router.post("/api/ema9/screen")
async def ema9_screen(req: Ema9ScreenRequest):
    tickers = [t.strip().upper() for t in req.tickers if t.strip()][:2000]
    if not tickers:
        raise HTTPException(400, "No tickers provided.")

    signals, filtered_by_trend, failed = [], [], []
    total    = len(tickers)
    n_chunks = math.ceil(total / YF_CHUNK_SIZE)

    # ── Step 1: Chunked batch yf.download ─────────────────────
    ticker_dfs: Dict[str, pd.DataFrame] = {}

    for chunk_idx in range(n_chunks):
        chunk = tickers[chunk_idx * YF_CHUNK_SIZE : (chunk_idx + 1) * YF_CHUNK_SIZE]
        chunk_dfs = await asyncio.to_thread(
            _batch_download, chunk, req.interval, req.lookback_days
        )
        ticker_dfs.update(chunk_dfs)
        if chunk_idx < n_chunks - 1:
            await asyncio.sleep(YF_CHUNK_DELAY)

    # ── Step 2: Process each ticker ───────────────────────────
    for ticker in tickers:
        if ticker not in ticker_dfs:
            failed.append({"ticker": ticker, "error": "No data from batch download"})
            continue
        try:
            res = _process_ticker_df(ticker, ticker_dfs[ticker], req.max_candles_ago)
            res["interval"] = req.interval

            if res["status"] in ("ERROR", "NO_DATA"):
                failed.append({"ticker": res["ticker"], "error": res.get("error", "")})
                continue

            if res["status"] != "SIGNAL":
                continue

            # ── Trend filter ─────────────────────────────────
            regime = res.get("trend_regime", "UNKNOWN")
            if req.require_uptrend:
                if regime == "DOWNTREND":
                    filtered_by_trend.append({
                        "ticker":       res["ticker"],
                        "trend_regime": regime,
                        "reason":       "Rejected: stock in DOWNTREND",
                    })
                    continue
                if not req.allow_sideways and regime == "SIDEWAYS":
                    filtered_by_trend.append({
                        "ticker":       res["ticker"],
                        "trend_regime": regime,
                        "reason":       "Rejected: stock in SIDEWAYS (filter active)",
                    })
                    continue

            signals.append(res)

        except Exception as exc:
            failed.append({"ticker": ticker, "error": str(exc)})

    # Sort: most recent confirmation first
    signals.sort(key=lambda r: r.get("candles_ago", 999))

    # ── Step 3: Fair Value enrichment ─────────────────────────
    _SAFE_FV_KEYS = {
        "composite_fair_price", "composite_gain_pct",
        "fair_gap_pct", "valuation_bucket", "fv_model_count",
    }

    for i, sig in enumerate(signals):
        fv = await asyncio.to_thread(_enrich_fair_value, sig["ticker"])
        for k in _SAFE_FV_KEYS:
            sig[k] = fv.get(k)
        if i < len(signals) - 1:
            await asyncio.sleep(FV_INTER_DELAY)

    # ── Step 4: Partition into Prime Targets vs Technical Only ─
    # Prime Target: Price < Fair Value (undervalued)
    # Technical Only: Price > Fair Value (overvalued)
    prime_targets = [
        s for s in signals
        if s.get("composite_fair_price") is not None
        and s["current_price"] < s["composite_fair_price"]
    ]
    technical_only = [
        s for s in signals
        if s.get("composite_fair_price") is not None
        and s["current_price"] >= s["composite_fair_price"]
    ]
    # Signals without fair value data
    no_fv_signals = [
        s for s in signals
        if s.get("composite_fair_price") is None
    ]

    return {
        "signals":             signals,           # ALL passing signals
        "prime_targets":       prime_targets,     # Price < FV
        "technical_only":      technical_only,    # Price >= FV
        "no_fv_signals":       no_fv_signals,     # No FV data
        "filtered_by_trend":   filtered_by_trend,
        "count":               len(signals),
        "prime_count":         len(prime_targets),
        "technical_count":     len(technical_only),
        "no_fv_count":         len(no_fv_signals),
        "failed":              failed,
        "interval":            req.interval,
    }
