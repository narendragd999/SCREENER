"""
9EMA Breakout Screener v3.3 — FastAPI Router
═════════════════════════════════════════════
Fixes applied IN ema9_router.py (without modifying sma_router.py):
  • Global yf.download monkey-patch: Fixes the future end-date bug (2027-03-31)
    that causes "possibly delisted" errors in sma_router.py's FV fetch.
  • Global yf.download monkey-patch: Adds .BO fallback for single-ticker calls.
  • Replaced yf.download batch mode with yf.Ticker().history() loop for
    rock-solid EMA9 price data downloads.
"""

import os, io, asyncio, datetime as dt_module, time, math, logging
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel

# ─────────────────────────────────────────────────────────────
#  YFINANCE GLOBAL PATCH (Fixes sma_router.py bugs from here)
# ─────────────────────────────────────────────────────────────
_original_yf_download = yf.download

def _safe_yf_download(tickers, start=None, end=None, **kwargs):
    """
    Intercepts yf.download calls globally.
    1. Clamps future end dates (e.g., 2027-03-31) to today+7 days.
    2. Adds .BO fallback for single-ticker string calls that return empty.
    """
    # Fix 1: Future end dates cause "possibly delisted" errors
    if end is not None:
        try:
            end_dt = pd.to_datetime(end)
            if end_dt > pd.Timestamp.now() + pd.Timedelta(days=30):
                end = str((pd.Timestamp.now() + pd.Timedelta(days=7)).date())
        except Exception:
            pass

    # Call original download
    result = _original_yf_download(tickers, start=start, end=end, **kwargs)
    
    # Fix 2: .BO fallback for single-ticker string calls
    if isinstance(tickers, str) and (result is None or result.empty):
        if tickers.endswith(".NS"):
            bo_ticker = tickers.replace(".NS", ".BO")
            try:
                result = _original_yf_download(bo_ticker, start=start, end=end, **kwargs)
            except Exception:
                pass
                
    return result

# Apply the patch globally. Now when sma_router.py calls yf.download, it uses our safe version!
yf.download = _safe_yf_download


try:
    from sma_router import _analyze_ticker as _sma_analyze_ticker
    _FV_AVAILABLE = True
except ImportError:
    _FV_AVAILABLE = False

router = APIRouter()

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger("ema9_router")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DATA_DIR = "data"
FNO_CSV  = "tickers.csv"
ALL_CSV  = "tickers_all.csv"
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
YF_CHUNK_SIZE    = 100
YF_CHUNK_DELAY   = 2.0
FV_INTER_DELAY   = 1.5
FV_MAX_RETRIES   = 2
FV_RETRY_DELAY   = 3.0
MIN_HISTORY_DAYS = 60
DOWNLOAD_PERIOD  = "2y"

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
#  ROBUST SINGLE-TICKER DOWNLOAD (.NS → .BO fallback)
# ─────────────────────────────────────────────────────────────
def _download_single_ticker(ticker: str, period: str = DOWNLOAD_PERIOD) -> Optional[pd.DataFrame]:
    """
    Download a single ticker using yf.Ticker().history().
    Much more reliable than yf.download batch mode.
    Tries .NS first, then .BO if that fails.
    """
    for suffix in [".NS", ".BO"]:
        try:
            yf_sym = f"{ticker}{suffix}"
            t = yf.Ticker(yf_sym)
            hist = t.history(period=period, auto_adjust=True)
            if hist is not None and not hist.empty and len(hist) >= 10:
                df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                df = df.dropna(how="all")
                if len(df) >= 10:
                    return df
        except Exception:
            continue
    return None


# ─────────────────────────────────────────────────────────────
#  RELIABLE BATCH DOWNLOAD (Loop instead of yf.download)
# ─────────────────────────────────────────────────────────────
def _batch_download(
    tickers: List[str],
    interval: str,
    lookback_days: int,
) -> Dict[str, pd.DataFrame]:
    """
    Download tickers individually using yf.Ticker().history().
    Avoids yf.download batch mode which randomly throws 'possibly delisted'.
    """
    if not tickers:
        return {}

    result: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = _download_single_ticker(ticker, period=DOWNLOAD_PERIOD)
        if df is not None:
            result[ticker] = df
        time.sleep(0.15)  # Small delay to avoid yfinance rate limiting

    return result


# ─────────────────────────────────────────────────────────────
#  TREND DETECTION
# ─────────────────────────────────────────────────────────────
def _compute_trend(df: pd.DataFrame) -> Dict:
    n = len(df)
    close = df["Close"]

    sma50  = float(close.rolling(50).mean().iloc[-1])  if n >= 50  else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else None
    price  = float(close.iloc[-1])

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
    df = df.copy().dropna()

    if len(df) < MIN_HISTORY_DAYS:
        return {
            "ticker":  ticker,
            "status":  "NO_DATA",
            "error":   f"Insufficient data ({len(df)} days, need ≥ {MIN_HISTORY_DAYS})",
        }

    df["ema9"]  = df["Close"].ewm(span=9, adjust=False).mean()
    df["sma50"] = df["Close"].rolling(50).mean()

    n             = len(df)
    current_price = round(float(df["Close"].iloc[-1]), 2)
    current_ema9  = round(float(df["ema9"].iloc[-1]),  2)
    current_sma50 = round(float(df["sma50"].iloc[-1]), 2) if not pd.isna(df["sma50"].iloc[-1]) else None

    above_sma50 = current_sma50 is not None and current_price > current_sma50
    target_3pct = round(current_price * 1.03, 2)
    trend_data  = _compute_trend(df)

    scan_start = max(1, n - max_candles_ago - 1)
    scan_end   = n - 2
    found        = False
    breakout_idx = None
    confirm_idx  = None

    for i in range(scan_end, scan_start - 1, -1):
        prev_close = float(df["Close"].iloc[i - 1])
        prev_ema   = float(df["ema9"].iloc[i - 1])
        curr_close = float(df["Close"].iloc[i])
        curr_ema   = float(df["ema9"].iloc[i])

        if not (prev_close < prev_ema and curr_close > curr_ema):
            continue

        conf_close = float(df["Close"].iloc[i + 1])
        if conf_close > curr_close:
            found        = True
            breakout_idx = i
            confirm_idx  = i + 1
            break

    if not found:
        return {
            "ticker":        ticker,
            "status":        "NO_SIGNAL",
            "current_price": current_price,
            "ema9":          current_ema9,
            "ema9_value":    current_ema9,
            "sma50_value":   current_sma50,
            "above_sma50":   above_sma50,
            "target_3pct":   target_3pct,
            **trend_data,
        }

    breakout_candle = df.iloc[breakout_idx]
    confirm_candle  = df.iloc[confirm_idx]

    def _date(idx):
        d = df.index[idx]
        return str(d.date()) if hasattr(d, "date") else str(d)[:10]

    bo_date  = _date(breakout_idx)
    con_date = _date(confirm_idx)

    entry_price       = float(confirm_candle["Close"])
    target_from_entry = entry_price * 1.03
    target_nearing    = current_price >= target_from_entry * 0.995

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
        "ticker":          ticker,
        "status":          "SIGNAL",
        "current_price":   current_price,
        "target_3pct":     target_3pct,
        "target_nearing":  target_nearing,
        "ema9":            current_ema9,
        "ema9_value":      current_ema9,
        "sma50_value":     current_sma50,
        "above_sma50":     above_sma50,
        "ema9_dist_pct":   round(abs(current_price - current_ema9) / current_ema9 * 100, 2),
        "breakout_date":   bo_date,
        "breakout_close":  round(float(breakout_candle["Close"]), 2),
        "breakout_high":   round(float(breakout_candle["High"]),  2),
        "confirm_date":    con_date,
        "confirm_close":   round(float(confirm_candle["Close"]), 2),
        "candles_ago":     n - 1 - confirm_idx,
        "interval":        "1d",
        "candles":         candles,
        **trend_data,
    }


# ─────────────────────────────────────────────────────────────
#  SINGLE-TICKER SCREEN
# ─────────────────────────────────────────────────────────────
def _screen_ticker(ticker: str, max_candles_ago: int = 10) -> Dict:
    ticker = ticker.strip().upper()
    df = _download_single_ticker(ticker)
    if df is None or len(df) < MIN_HISTORY_DAYS:
        return {
            "ticker":  ticker,
            "status":  "NO_DATA",
            "error":   f"Insufficient data (need ≥ {MIN_HISTORY_DAYS} days)",
        }
    return _process_ticker_df(ticker, df, max_candles_ago)


# ─────────────────────────────────────────────────────────────
#  FAIR VALUE ENRICHMENT (Now fixed by the global yf.download patch)
# ─────────────────────────────────────────────────────────────
_FV_NULL = {
    "composite_fair_price": None,
    "composite_gain_pct":   None,
    "fair_gap_pct":         None,
    "valuation_bucket":     "N/A",
    "fv_model_count":       0,
    "gap_to_fair_pct":      None,
    "fv_error":             "NOT_AVAILABLE",
}

_SAFE_FV_KEYS = {
    "composite_fair_price", "composite_gain_pct",
    "fair_gap_pct", "valuation_bucket", "fv_model_count",
    "gap_to_fair_pct", "fv_error",
}


def _enrich_fair_value(ticker: str, current_price: float = None) -> Dict:
    if not _FV_AVAILABLE:
        logger.warning(f"[FV] {ticker}: sma_router not available (import failed)")
        return {**_FV_NULL, "fv_error": "SMA_ROUTER_NOT_AVAILABLE"}

    last_error = None

    for attempt in range(1, FV_MAX_RETRIES + 1):
        try:
            # This calls sma_router._analyze_ticker.
            # Because of our global monkey-patch at the top of this file,
            # the yf.download call inside sma_router will be safely intercepted!
            res = _sma_analyze_ticker(
                ticker,
                fy_start=2014,
                force=False,
                include_other_income=True,
            )

            if "error" in res:
                last_error = res["error"]
                logger.warning(f"[FV] {ticker} attempt {attempt}: sma_router returned error: {last_error}")
                if attempt < FV_MAX_RETRIES:
                    time.sleep(FV_RETRY_DELAY)
                    continue
                return {**_FV_NULL, "fv_error": f"SMA_ERROR: {str(last_error)[:120]}"}

            comp_fair = res.get("composite_fair_price")
            comp_gain = res.get("composite_gain_pct")
            bucket    = res.get("valuation_bucket", "N/A")
            model_cnt = res.get("model_count", 0)

            if comp_fair is None or comp_gain is None:
                logger.warning(f"[FV] {ticker} attempt {attempt}: No composite_fair_price in response.")
                if attempt < FV_MAX_RETRIES:
                    time.sleep(FV_RETRY_DELAY)
                    continue
                return {**_FV_NULL, "fv_error": "NO_COMPOSITE_FV_IN_RESPONSE"}

            gap_to_fair = None
            if current_price and current_price > 0:
                gap_to_fair = round((comp_fair - current_price) / current_price * 100, 2)

            logger.info(f"[FV] {ticker}: FV=₹{comp_fair:.1f}, Gap={gap_to_fair}%, Bucket={bucket}, Models={model_cnt}")

            return {
                "composite_fair_price": comp_fair,
                "composite_gain_pct":   comp_gain,
                "fair_gap_pct":         comp_gain,
                "valuation_bucket":     bucket,
                "fv_model_count":       model_cnt,
                "gap_to_fair_pct":      gap_to_fair,
                "fv_error":             None,
            }

        except Exception as exc:
            last_error = str(exc)
            logger.error(f"[FV] {ticker} attempt {attempt}: Exception: {exc}")
            if attempt < FV_MAX_RETRIES:
                time.sleep(FV_RETRY_DELAY)
                continue

    return {**_FV_NULL, "fv_error": f"RETRY_EXHAUSTED: {str(last_error)[:120]}"}


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
#  SHARED SCREENING PIPELINE
# ─────────────────────────────────────────────────────────────
async def _run_screen_pipeline(
    tickers: List[str],
    interval: str = "1d",
    lookback_days: int = 180,
    max_candles_ago: int = 10,
    require_uptrend: bool = True,
) -> Dict:
    tickers = [
        t.strip().upper().replace(".NS", "").replace(".BO", "")
        for t in tickers if t.strip()
    ][:2000]

    if not tickers:
        raise HTTPException(400, "No valid tickers provided.")

    signals, filtered_by_trend, failed = [], [], []

    # Step 1: Batch download (using robust yf.Ticker loop)
    ticker_dfs = await asyncio.to_thread(_batch_download, tickers, interval, lookback_days)

    # Step 2: Process each ticker
    for ticker in tickers:
        if ticker not in ticker_dfs:
            failed.append({"ticker": ticker, "error": "No data available (tried .NS and .BO)"})
            continue
        try:
            res = _process_ticker_df(ticker, ticker_dfs[ticker], max_candles_ago)
            res["interval"] = interval

            if res["status"] in ("ERROR", "NO_DATA"):
                failed.append({"ticker": res["ticker"], "error": res.get("error", "")})
                continue

            if res["status"] != "SIGNAL":
                continue

            if require_uptrend and not res.get("above_sma50", False):
                filtered_by_trend.append({
                    "ticker":       res["ticker"],
                    "trend_regime": res.get("trend_regime", "UNKNOWN"),
                    "reason":       "Rejected: Close ≤ 50-day SMA (no uptrend)",
                })
                continue

            signals.append(res)

        except Exception as exc:
            failed.append({"ticker": ticker, "error": str(exc)})

    signals.sort(key=lambda r: r.get("candles_ago", 999))

    # Step 3: Fair Value enrichment
    fv_failures = []
    for i, sig in enumerate(signals):
        fv = await asyncio.to_thread(
            _enrich_fair_value, sig["ticker"], sig.get("current_price")
        )
        for k in _SAFE_FV_KEYS:
            sig[k] = fv.get(k)

        if fv.get("fv_error"):
            fv_failures.append({"ticker": sig["ticker"], "fv_error": fv["fv_error"]})

        if i < len(signals) - 1:
            await asyncio.sleep(FV_INTER_DELAY)

    if fv_failures:
        logger.info(f"[FV] {len(fv_failures)}/{len(signals)} tickers had FV errors: "
                     f"{[f['ticker'] for f in fv_failures]}")

    # Step 4: Partition
    prime_targets = []
    other_signals = []

    for sig in signals:
        fv = sig.get("composite_fair_price")
        cp = sig.get("current_price")
        if fv and cp and cp < fv:
            prime_targets.append(sig)
        else:
            other_signals.append(sig)

    return {
        "signals":             signals,
        "prime_targets":       prime_targets,
        "other_signals":       other_signals,
        "undervalued_signals": prime_targets,
        "filtered_by_trend":   filtered_by_trend,
        "failed":              failed,
        "fv_failures":         fv_failures,
        "count":               len(signals),
        "prime_count":         len(prime_targets),
        "other_count":         len(other_signals),
        "undervalued_count":   len(prime_targets),
        "interval":            interval,
    }


# ─────────────────────────────────────────────────────────────
#  CSV PARSING HELPER
# ─────────────────────────────────────────────────────────────
def _parse_csv_tickers(content: bytes) -> Dict:
    df = pd.read_csv(io.BytesIO(content), encoding="utf-8-sig")

    symbol_col = None
    _TARGET_NAMES = {"symbol", "ticker", "stock", "ticker symbol", "stock symbol"}
    for col in df.columns:
        if col.strip().lower() in _TARGET_NAMES:
            symbol_col = col
            break

    if symbol_col is None:
        raise HTTPException(
            400,
            f"No Symbol/Ticker/Stock column found. Available: {list(df.columns)}",
        )

    tickers = (
        df[symbol_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".NS", "", regex=False)
        .str.replace(".BO", "", regex=False)
    )
    tickers = tickers[tickers != ""].unique().tolist()

    return {"tickers": tickers, "total_rows": len(df), "found_column": symbol_col}


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
    return await _run_screen_pipeline(
        tickers=req.tickers,
        interval=req.interval,
        lookback_days=req.lookback_days,
        max_candles_ago=req.max_candles_ago,
        require_uptrend=req.require_uptrend,
    )


@router.post("/api/ema9/upload-csv")
async def ema9_upload_csv(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only .csv files are accepted.")

    content = await file.read()
    if len(content) > 10 * 1024 * 1024:
        raise HTTPException(413, "File too large (max 10 MB).")

    csv_info = _parse_csv_tickers(content)
    tickers  = csv_info["tickers"]

    if not tickers:
        raise HTTPException(400, "No valid tickers found in CSV.")

    result = await _run_screen_pipeline(
        tickers=tickers,
        interval="1d",
        lookback_days=180,
        max_candles_ago=10,
        require_uptrend=True,
    )

    result["csv_info"] = {
        "filename":       file.filename,
        "total_rows":     csv_info["total_rows"],
        "parsed_tickers": len(tickers),
        "found_column":   csv_info["found_column"],
    }

    return result


@router.get("/api/ema9/quick-scan/{ticker}")
async def ema9_quick_scan(ticker: str):
    ticker = ticker.strip().upper().replace(".NS", "").replace(".BO", "")

    if not ticker or not ticker.isalnum():
        raise HTTPException(400, "Invalid ticker symbol.")

    result = await asyncio.to_thread(_screen_ticker, ticker)
    result["interval"] = "1d"

    fv = await asyncio.to_thread(
        _enrich_fair_value, ticker, result.get("current_price")
    )
    for k in _SAFE_FV_KEYS:
        result[k] = fv.get(k)

    if result.get("status") == "SIGNAL":
        fv_price = result.get("composite_fair_price")
        cp       = result.get("current_price")
        if fv_price and cp and cp < fv_price:
            result["signal_type"] = "PRIME"
        else:
            result["signal_type"] = "TECHNICAL_ONLY"
    elif result.get("status") == "NO_SIGNAL":
        result["signal_type"] = "NO_SIGNAL"
    else:
        result["signal_type"] = result.get("status", "UNKNOWN")

    return result


@router.get("/api/ema9/fair-value/{ticker}")
async def ema9_fair_value(ticker: str):
    ticker = ticker.strip().upper().replace(".NS", "").replace(".BO", "")

    if not ticker or not ticker.isalnum():
        raise HTTPException(400, "Invalid ticker symbol.")

    if not _FV_AVAILABLE:
        raise HTTPException(503, "Fair Value engine (sma_router) is not available.")

    try:
        fv = await asyncio.to_thread(_enrich_fair_value, ticker, None)
    except Exception as exc:
        logger.error(f"[FV-OnDemand] {ticker}: {exc}")
        raise HTTPException(500, f"FV fetch failed: {str(exc)[:200]}")

    if fv.get("composite_fair_price") is None:
        error_msg = fv.get("fv_error", "UNKNOWN")
        logger.warning(f"[FV-OnDemand] {ticker}: No FV available. Error: {error_msg}")
        return {
            "ticker":  ticker,
            "status":  "NO_FV",
            "fv_error": error_msg,
            **fv,
        }

    return {
        "ticker":  ticker,
        "status":  "OK",
        **fv, 
    }