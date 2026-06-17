import os, time, asyncio, threading, math
from datetime import datetime, timedelta
from typing import Optional, List

import pandas as pd
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

# ─────────────────────────────────────────────────────────────
#  NSE SESSION (isolated from main.py's session)
# ─────────────────────────────────────────────────────────────
_oc_session: Optional[requests.Session] = None
_oc_lock    = threading.Lock()
_oc_warmed  = False

# API URLs
NSE_OC_URL      = "https://www.nseindia.com/api/historicalOR/foCPV"
NSE_EQUITY_URL  = "https://www.nseindia.com/api/historicalOR/generateSecurityWiseHistoricalData"

INSTRUMENT_TYPES = ["OPTSTK", "OPTIDX", "FUTIDX", "FUTSTK", "FUTIVX"]
OPTION_TYPES     = ["CE", "PE"]
QUICK_RANGES     = ["Custom", "1D", "1W", "1M", "1.5M", "3M"]


def _get_oc_session() -> requests.Session:
    global _oc_session
    with _oc_lock:
        if _oc_session is not None:
            return _oc_session
        s = requests.Session()
        s.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept":                    "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Accept-Language":           "en-US,en;q=0.9",
            "Accept-Encoding":           "gzip, deflate",
            "Connection":                "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        })
        _oc_session = s
        return s


def _reset_oc_session():
    global _oc_session, _oc_warmed
    with _oc_lock:
        _oc_session = None
        _oc_warmed  = False


def _warm_up(session: requests.Session) -> bool:
    global _oc_warmed
    if _oc_warmed:
        return True
    warm_pages = [
        "https://www.nseindia.com/",
        "https://www.nseindia.com/market-data/equity-derivatives-watch",
    ]
    for page in warm_pages:
        try:
            session.get(page, timeout=15)
            time.sleep(1.5)
        except Exception:
            return False
    _oc_warmed = True
    return True


# ════════════════════════════════════════════════════════════════════
#  🆕 NEW: UNDERLYING PRICE FETCHER — EQUITY HISTORICAL DATA API
# ════════════════════════════════════════════════════════════════════
def _fetch_underlying_price_equity(
    symbol: str,
    target_date: datetime,
) -> Optional[dict]:
    """
    Fetch underlying equity price using the NEW NSE Historical Data API.
    
    Endpoint: /api/historicalOR/generateSecurityWiseHistoricalData
    Parameters: from, to, symbol, type=priceVolumeDeliverable, series=ALL
    
    Returns dict with keys: price, date, high, low, open, close, volume
    or None if data not available.
    """
    session = _get_oc_session()
    _warm_up(session)

    # Format dates - search within ±5 days of target for better hit rate
    from_dt = target_date - timedelta(days=5)
    to_dt   = target_date + timedelta(days=5)
    
    params = {
        "from":   from_dt.strftime("%d-%m-%Y"),
        "to":     to_dt.strftime("%d-%m-%Y"),
        "symbol": symbol.strip().upper(),
        "type":   "priceVolumeDeliverable",
        "series": "ALL",
    }
    
    api_hdrs = {
        "Accept":           "application/json, text/plain, */*",
        "Referer":          "https://www.nseind.com/market-data/equities-historical",
        "X-Requested-With": "XMLHttpRequest",
    }

    for attempt in range(2):
        try:
            r = session.get(NSE_EQUITY_URL, params=params, headers=api_hdrs, timeout=15)
            
            if r.status_code == 401:
                _warm_up(session)
                time.sleep(2)
                continue
            
            if r.status_code == 403:
                _reset_oc_session()
                return None
                
            if r.status_code != 200:
                return None

            resp_data = r.json()
            if not resp_data or "data" not in resp_data or not resp_data["data"]:
                return None

            # Find the row closest to target_date
            best_match = None
            best_diff  = None
            target_str = target_date.strftime("%d %b %Y")  # NSE format like "12 Dec 2022"
            
            for row in resp_data["data"]:
                ts = row.get("mTIMESTAMP") or row.get("timestamp") or row.get("date")
                if not ts:
                    continue
                    
                try:
                    # Try multiple date formats NSE might return
                    for fmt in ["%d %b %Y", "%d-%b-%Y", "%Y-%m-%d"]:
                        try:
                            row_date = datetime.strptime(str(ts).strip(), fmt)
                            diff = abs((row_date.date() - target_date.date()).days)
                            if best_diff is None or diff < best_diff:
                                best_diff = diff
                                best_match = row
                            break
                        except ValueError:
                            continue
                except Exception:
                    continue

            if best_match and best_diff is not None and best_diff <= 5:
                # Extract price fields - NSE equity API field names
                price_data = {
                    "price":  float(best_match.get("CH_CLOSING_PRICE") or best_match.get("CH_TRADE_LAST_PRICE") or 0),
                    "date":   str(best_match.get("mTIMESTAMP") or best_match.get("timestamp") or ""),
                    "open":   float(best_match.get("CH_OPENING_PRICE") or 0),
                    "high":   float(best_match.get("CH_TRADE_HIGH_PRICE") or 0),
                    "low":    float(best_match.get("CH_TRADE_LOW_PRICE") or 0),
                    "close":  float(best_match.get("CH_CLOSING_PRICE") or 0),
                    "volume": int(best_match.get("CH_TOTAL_TRADED_QUANTITY") or 0),
                    "source": "equity_api",
                }
                
                # Validate we got a real price
                if price_data["price"] > 0:
                    return price_data
                    
            return None

        except requests.RequestException:
            if attempt == 1:
                return None
            time.sleep(2)
        except Exception:
            return None

    return None


def _fetch_underlying_price_fallback(
    symbol: str,
    instrument_type: str,
    expiry_dt: datetime,
    option_type: str,
    entry_dt: datetime,
) -> Optional[float]:
    """
    Fallback: Try to get underlying price from options data (original method).
    Uses FH_UNDERLYING_VALUE from foCPV API.
    """
    session = _get_oc_session()
    _warm_up(session)
    
    api_hdrs = {
        "Accept":           "application/json, text/plain, */*",
        "Referer":          "https://www.nseindia.com/market-data/equity-derivatives-watch",
        "X-Requested-With": "XMLHttpRequest",
    }

    # Try wide window around entry date
    entry_from = entry_dt - timedelta(days=5)
    entry_to   = min(entry_dt + timedelta(days=10), expiry_dt)
    
    params = {
        "from":           entry_from.strftime("%d-%m-%Y"),
        "to":             entry_to.strftime("%d-%m-%Y"),
        "instrumentType": instrument_type,
        "symbol":         symbol,
        "year":           str(expiry_dt.year),
        "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
        "optionType":     option_type,
    }

    try:
        r = session.get(NSE_OC_URL, params=params, headers=api_hdrs, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if data and "data" in data and data["data"]:
                best_diff = None
                best_price = None
                for row in data["data"]:
                    uv = row.get("FH_UNDERLYING_VALUE")
                    ts = row.get("FH_TIMESTAMP")
                    if uv and ts:
                        try:
                            uval = float(uv)
                            row_dt = datetime.strptime(ts, "%d-%b-%Y")
                            diff = abs((row_dt.date() - entry_dt.date()).days)
                            if best_diff is None or diff < best_diff:
                                best_diff = diff
                                best_price = uval
                        except Exception:
                            pass
                if best_price:
                    return {"price": best_price, "source": "options_fallback"}
    except Exception:
        pass
    
    return None


def _get_underlying_price(
    symbol: str,
    target_date: datetime,
    instrument_type: str = "OPTSTK",
    expiry_dt: Optional[datetime] = None,
    option_type: str = "CE",
) -> Optional[dict]:
    """
    Unified underlying price fetcher.
    
    Strategy:
    1. PRIMARY: Use new Equity Historical Data API (more reliable for spot prices)
    2. FALLBACK: Extract from options OHLC data (FH_UNDERLYING_VALUE)
    
    Returns dict with 'price', 'date', 'source' keys or None.
    """
    # ── Primary: Equity API ──
    result = _fetch_underlying_price_equity(symbol, target_date)
    if result and result.get("price"):
        return result
    
    # ── Fallback: Options API ──
    if expiry_dt:
        result = _fetch_underlying_price_fallback(
            symbol, instrument_type, expiry_dt, option_type, target_date
        )
        if result and result.get("price"):
            return result
    
    return None


# ─────────────────────────────────────────────────────────────
#  CORE FETCH — OHLC DATA (blocking)
# ─────────────────────────────────────────────────────────────
def _do_fetch(
    from_dt: datetime, to_dt: datetime,
    symbol: str, year: int, expiry_dt: datetime,
    option_type: str, strike_price: int, instrument_type: str,
) -> list:
    session = _get_oc_session()
    _warm_up(session)

    params = {
        "from":           from_dt.strftime("%d-%m-%Y"),
        "to":             to_dt.strftime("%d-%m-%Y"),
        "instrumentType": instrument_type,
        "symbol":         symbol,
        "year":           str(year),
        "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
        "optionType":     option_type,
        "strikePrice":    str(strike_price),
    }
    api_hdrs = {
        "Accept":           "application/json, text/plain, */*",
        "Referer":          "https://www.nseindia.com/market-data/equity-derivatives-watch",
        "X-Requested-With": "XMLHttpRequest",
    }

    resp_data = None
    for attempt in range(2):
        try:
            r = session.get(NSE_OC_URL, params=params, headers=api_hdrs, timeout=15)
            if r.status_code == 401:
                _warm_up(session)
                time.sleep(3)
                continue
            if r.status_code == 403:
                _reset_oc_session()
                raise ValueError("HTTP 403 — NSE blocked the request. Try again after a few seconds.")
            if r.status_code != 200:
                raise ValueError(f"HTTP {r.status_code} from NSE.")
            resp_data = r.json()
            break
        except ValueError:
            raise
        except requests.RequestException as exc:
            if attempt == 1:
                raise ValueError(f"Network error: {exc}")
            time.sleep(3)

    if not resp_data or "data" not in resp_data or not resp_data["data"]:
        raise ValueError(
            "No data returned by NSE. Verify: symbol exists in F&O, "
            "expiry date is correct, strike price is valid for that expiry."
        )

    df = pd.DataFrame(resp_data["data"])

    rename_map = {
        "FH_TIMESTAMP":         "date",
        "FH_OPENING_PRICE":     "open",
        "FH_TRADE_HIGH_PRICE":  "high",
        "FH_TRADE_LOW_PRICE":   "low",
        "FH_CLOSING_PRICE":     "close",
        "FH_LAST_TRADED_PRICE": "ltp",
        "FH_STRIKE_PRICE":      "strike_price",
        "FH_EXPIRY_DT":         "expiry",
        "FH_OPTION_TYPE":       "option_type_col",
        "FH_UNDERLYING_VALUE":  "underlying",
        "FH_TOT_TRADED_QTY":    "volume",
        "FH_OPEN_INT":          "oi",
        "FH_CHG_IN_OI":         "change_oi",
        "FH_SETTLE_PRICE":      "settle_price",
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    numeric = ["open", "high", "low", "close", "ltp", "volume", "oi", "change_oi",
               "underlying", "settle_price"]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace("-", None, inplace=True)
    df.dropna(subset=["open", "high", "low", "close"], inplace=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], format="%d-%b-%Y", errors="coerce")
        df.sort_values("date", inplace=True)
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    for col in ["open", "high", "low", "close", "ltp", "underlying"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    def _safe(v):
        if v is None:
            return None
        try:
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
        except Exception:
            pass
        return v

    records = df.to_dict(orient="records")
    clean   = [{k: _safe(v) for k, v in row.items()} for row in records]
    return clean


# ─────────────────────────────────────────────────────────────
#  STRIKES FETCH
# ─────────────────────────────────────────────────────────────
def _do_fetch_strikes(
    symbol: str,
    instrument_type: str,
    expiry_dt: datetime,
    option_type: str,
) -> list:
    session = _get_oc_session()
    _warm_up(session)

    today = datetime.now().date()
    if expiry_dt.date() >= today:
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=7)
    else:
        to_dt   = expiry_dt
        from_dt = expiry_dt - timedelta(days=3)

    params = {
        "from":           from_dt.strftime("%d-%m-%Y"),
        "to":             to_dt.strftime("%d-%m-%Y"),
        "instrumentType": instrument_type,
        "symbol":         symbol,
        "year":           str(expiry_dt.year),
        "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
        "optionType":     option_type,
    }
    api_hdrs = {
        "Accept":           "application/json, text/plain, */*",
        "Referer":          "https://www.nseindia.com/market-data/equity-derivatives-watch",
        "X-Requested-With": "XMLHttpRequest",
    }

    for attempt in range(2):
        try:
            r = session.get(NSE_OC_URL, params=params, headers=api_hdrs, timeout=15)
            if r.status_code == 401:
                _warm_up(session)
                time.sleep(2)
                continue
            if r.status_code == 403:
                return []
            if r.status_code != 200:
                return []

            data = r.json()
            if not data or "data" not in data or not data["data"]:
                return []

            strikes = set()
            for row in data["data"]:
                strike_val = row.get("FH_STRIKE_PRICE")
                if strike_val is not None:
                    try:
                        s = int(float(strike_val))
                        if s > 0:
                            strikes.add(s)
                    except (ValueError, TypeError):
                        pass

            return sorted(strikes)

        except requests.RequestException:
            if attempt == 1:
                return []
            time.sleep(2)
        except Exception:
            return []

    return []


# ─────────────────────────────────────────────────────────────
#  BEAR CALL SPREAD — CORE CALCULATION ENGINE
# ─────────────────────────────────────────────────────────────

def _build_daily_pnl_series(
    short_data: list,      
    long_data: list,       
    entry_date: str,       
    short_entry_premium: float,   
    long_entry_premium: float,    
    lot_size: int,
    num_lots: int,
) -> list:
    """Build day-by-day P&L for a Bear Call Spread."""
    short_map = {r["date"]: r for r in short_data}
    long_map  = {r["date"]: r for r in long_data}

    all_dates = sorted(set(short_map.keys()) | set(long_map.keys()))
    all_dates = [d for d in all_dates if d >= entry_date]

    net_credit_per_share = round(short_entry_premium - long_entry_premium, 2)
    max_profit_per_lot   = net_credit_per_share * lot_size
    total_max_profit     = max_profit_per_lot * num_lots

    rows = []
    for d in all_dates:
        sr = short_map.get(d)
        lr = long_map.get(d)
        if sr is None or lr is None:
            continue

        sc = sr.get("close") or sr.get("ltp")
        lc = lr.get("close") or lr.get("ltp")
        if sc is None or lc is None:
            continue

        pnl_per_share  = net_credit_per_share - (sc - lc)
        pnl_per_lot    = round(pnl_per_share * lot_size, 2)
        pnl_total      = round(pnl_per_lot * num_lots, 2)
        pct_of_max     = round((pnl_total / total_max_profit * 100) if total_max_profit != 0 else 0, 1)

        underlying = sr.get("underlying")

        rows.append({
            "date":            d,
            "short_close":     round(sc, 2),
            "long_close":      round(lc, 2),
            "pnl_per_share":   round(pnl_per_share, 2),
            "pnl_per_lot":     pnl_per_lot,
            "pnl_total":       pnl_total,
            "pct_of_max":      pct_of_max,
            "underlying":      round(underlying, 2) if underlying else None,
        })

    return rows


def _calc_spread_stats(
    pnl_rows: list,
    short_strike: int,
    long_strike: int,
    short_entry: float,
    long_entry: float,
    lot_size: int,
    num_lots: int,
    expiry_str: str,         
) -> dict:
    """Calculate summary statistics for the spread."""
    if not pnl_rows:
        return {}

    net_credit      = round(short_entry - long_entry, 2)
    spread_width    = long_strike - short_strike
    max_profit_ps   = net_credit
    max_loss_ps     = round(spread_width - net_credit, 2)
    breakeven       = round(short_strike + net_credit, 2)

    total_lots      = num_lots
    max_profit_tot  = round(max_profit_ps * lot_size * total_lots, 2)
    max_loss_tot    = round(max_loss_ps * lot_size * total_lots, 2)
    capital_at_risk = max_loss_tot

    final_row   = pnl_rows[-1]
    final_pnl   = final_row["pnl_total"]
    final_pct   = final_row["pct_of_max"]

    pnl_series  = [r["pnl_total"] for r in pnl_rows]
    peak_pnl    = max(pnl_series)
    trough_pnl  = min(pnl_series)
    max_dd      = round(peak_pnl - trough_pnl, 2) if peak_pnl > trough_pnl else 0

    days_profit = sum(1 for p in pnl_series if p > 0)
    days_loss   = sum(1 for p in pnl_series if p < 0)
    days_flat   = len(pnl_series) - days_profit - days_loss

    rr = round(max_profit_tot / max_loss_tot, 3) if max_loss_tot != 0 else None

    if final_pnl >= max_profit_tot * 0.95:
        outcome = "FULL PROFIT 🎯"
    elif final_pnl > 0:
        outcome = "PARTIAL PROFIT ✅"
    elif final_pnl == 0:
        outcome = "BREAKEVEN ⚖️"
    elif final_pnl < 0 and final_pnl > max_loss_tot * 0.5:
        outcome = "PARTIAL LOSS ⚠️"
    else:
        outcome = "FULL LOSS ❌"

    return {
        "expiry":           expiry_str,
        "short_strike":     short_strike,
        "long_strike":      long_strike,
        "spread_width":     spread_width,
        "net_credit":       net_credit,
        "breakeven":        breakeven,
        "max_profit_ps":    max_profit_ps,
        "max_loss_ps":      max_loss_ps,
        "max_profit_total": max_profit_tot,
        "max_loss_total":   max_loss_tot,
        "capital_at_risk":  capital_at_risk,
        "risk_reward":      rr,
        "final_pnl":        final_pnl,
        "final_pct_of_max": final_pct,
        "peak_pnl":         round(peak_pnl, 2),
        "trough_pnl":       round(trough_pnl, 2),
        "max_drawdown":     max_dd,
        "days_total":       len(pnl_series),
        "days_profit":      days_profit,
        "days_loss":        days_loss,
        "days_flat":        days_flat,
        "outcome":          outcome,
    }


def _build_payoff_curve(
    short_strike: int,
    long_strike: int,
    net_credit: float,
    lot_size: int,
    num_lots: int,
    underlying_price: Optional[float] = None,
) -> dict:
    """Build at-expiry payoff curve data for the Bear Call Spread."""
    spread_width = long_strike - short_strike
    max_profit   = net_credit
    max_loss     = spread_width - net_credit
    breakeven    = short_strike + net_credit

    lo = int(short_strike * 0.85)
    hi = int(long_strike  * 1.15)
    step = max(1, (hi - lo) // 200)

    prices, pnls_ps, pnls_total = [], [], []
    for price in range(lo, hi + step, step):
        short_pnl = net_credit - max(0, price - short_strike)
        long_pnl  = max(0, price - long_strike)
        pnl_ps    = round(short_pnl + long_pnl, 4)
        pnl_ps    = max(-max_loss, min(max_profit, pnl_ps))
        prices.append(price)
        pnls_ps.append(pnl_ps)
        pnls_total.append(round(pnl_ps * lot_size * num_lots, 2))

    return {
        "prices":      prices,
        "pnl_ps":      pnls_ps,
        "pnl_total":   pnls_total,
        "breakeven":   round(breakeven, 2),
        "max_profit":  max_profit,
        "max_loss":    max_loss,
        "underlying":  underlying_price,
    }


def _do_bear_call_spread(
    symbol: str,
    instrument_type: str,
    short_strike: int,
    long_strike: int,
    expiry_dt: datetime,
    entry_date: str,         
    lot_size: int,
    num_lots: int,
) -> dict:
    """
    Full Bear Call Spread analysis for one expiry.
    UPDATED: Now uses new equity API for underlying price on entry date.
    """
    fmt = "%d-%m-%Y"
    from_dt = datetime.strptime(entry_date, "%Y-%m-%d")

    # Fetch short (sold) call leg
    short_rows = _do_fetch(
        from_dt, expiry_dt,
        symbol, expiry_dt.year, expiry_dt,
        "CE", short_strike, instrument_type,
    )
    # Fetch long (bought) call leg
    long_rows = _do_fetch(
        from_dt, expiry_dt,
        symbol, expiry_dt.year, expiry_dt,
        "CE", long_strike, instrument_type,
    )

    if not short_rows:
        raise ValueError(f"No data for short leg (CE {short_strike}) on expiry {expiry_dt.strftime('%d-%b-%Y')}")
    if not long_rows:
        raise ValueError(f"No data for long leg (CE {long_strike}) on expiry {expiry_dt.strftime('%d-%b-%Y')}")

    # Entry premiums = close/ltp on entry_date
    def _find_entry_premium(rows, entry):
        for r in sorted(rows, key=lambda x: x["date"]):
            if r["date"] >= entry:
                v = r.get("close") or r.get("ltp")
                if v:
                    return float(v), r["date"]
        return None, None

    short_entry_prem, short_entry_actual = _find_entry_premium(short_rows, entry_date)
    long_entry_prem,  long_entry_actual  = _find_entry_premium(long_rows,  entry_date)

    if short_entry_prem is None:
        raise ValueError(f"Could not determine entry premium for short leg (CE {short_strike})")
    if long_entry_prem is None:
        raise ValueError(f"Could not determine entry premium for long leg (CE {long_strike})")

    actual_entry_date = max(short_entry_actual, long_entry_actual)

    # Build daily P&L series
    pnl_rows = _build_daily_pnl_series(
        short_rows, long_rows,
        actual_entry_date,
        short_entry_prem, long_entry_prem,
        lot_size, num_lots,
    )

    # Stats
    stats = _calc_spread_stats(
        pnl_rows,
        short_strike, long_strike,
        short_entry_prem, long_entry_prem,
        lot_size, num_lots,
        expiry_dt.strftime("%d-%b-%Y"),
    )

    # Payoff curve
    net_credit    = short_entry_prem - long_entry_prem
    last_underlying = None
    if pnl_rows:
        last_underlying = pnl_rows[-1].get("underlying")
    payoff = _build_payoff_curve(
        short_strike, long_strike, net_credit,
        lot_size, num_lots, last_underlying,
    )

    # 🆕 UPDATED: Get underlying price using new unified method
    entry_underlying = None
    entry_underlying_date = None
    
    # First try the new equity API for entry date
    entry_dt_for_lookup = datetime.strptime(actual_entry_date, "%Y-%m-%d")
    ul_result = _get_underlying_price(symbol, entry_dt_for_lookup, instrument_type, expiry_dt, "CE")
    
    if ul_result and ul_result.get("price"):
        entry_underlying      = ul_result["price"]
        entry_underlying_date = ul_result.get("date", actual_entry_date)
    
    # Fallback: scan short_rows for closest to actual_entry_date  
    if entry_underlying is None:
        for row in short_rows:
            if row.get("date") == actual_entry_date:
                entry_underlying      = row.get("underlying")
                entry_underlying_date = row.get("date")
                break
        if entry_underlying is None and short_rows:
            for row in sorted(short_rows, key=lambda x: x["date"]):
                if row.get("date") >= actual_entry_date and row.get("underlying") is not None:
                    entry_underlying      = row.get("underlying")
                    entry_underlying_date = row.get("date")
                    break

    # OTM % for both strikes relative to underlying at entry
    short_otm_pct = None
    long_otm_pct  = None
    if entry_underlying and entry_underlying > 0:
        short_otm_pct = round((short_strike - entry_underlying) / entry_underlying * 100, 2)
        long_otm_pct  = round((long_strike  - entry_underlying) / entry_underlying * 100, 2)

    return {
        "entry_date":           actual_entry_date,
        "short_entry_premium":  round(short_entry_prem, 2),
        "long_entry_premium":   round(long_entry_prem, 2),
        "entry_underlying":     round(entry_underlying, 2) if entry_underlying else None,
        "entry_underlying_date": entry_underlying_date,
        "short_otm_pct":        short_otm_pct,
        "long_otm_pct":         long_otm_pct,
        "underlying_source":    ul_result.get("source") if ul_result else "options_data",
        "stats":                stats,
        "pnl_series":           pnl_rows,
        "payoff":               payoff,
        "short_ohlc":           short_rows,
        "long_ohlc":            long_rows,
    }


# ─────────────────────────────────────────────────────────────
#  PYDANTIC MODELS
# ─────────────────────────────────────────────────────────────
class OcFetchRequest(BaseModel):
    symbol:          str
    instrument_type: str = "OPTSTK"
    expiry_date:     str            
    option_type:     str = "CE"
    strike_price:    int
    from_date:       str            
    to_date:         str            

# 🆕 NEW MODEL for underlying price request
class UnderlyingPriceRequest(BaseModel):
    symbol:          str
    target_date:     str              # DD-MM-YYYY
    instrument_type: str = "OPTSTK"   # For fallback only
    expiry_date:     Optional[str] = None  # DD-MM-YYYY - For fallback only

class BearCallSpreadRequest(BaseModel):
    symbol:          str
    instrument_type: str = "OPTSTK"
    short_strike:    int            
    long_strike:     int            
    entry_date:      str            
    expiry_date_1:   str            
    expiry_date_2:   Optional[str] = None  
    lot_size:        int = 1        
    num_lots:        int = 1        

# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@router.get("/api/option-charts/meta")
async def oc_meta():
    tickers = []
    try:
        if os.path.exists("tickers.csv"):
            df = pd.read_csv("tickers.csv")
            if "SYMBOL" in df.columns:
                tickers = sorted(df["SYMBOL"].dropna().str.strip().str.upper().unique().tolist())
    except Exception:
        pass
    return {
        "tickers":          tickers,
        "instrument_types": INSTRUMENT_TYPES,
        "option_types":     OPTION_TYPES,
        "quick_ranges":     QUICK_RANGES,
    }


@router.get("/api/option-charts/strikes")
async def oc_strikes(
    symbol:          str,
    instrument_type: str = "OPTSTK",
    expiry_date:     str = "",
    option_type:     str = "CE",
):
    if not symbol or not expiry_date:
        return {"strikes": [], "source": "missing_params", "count": 0}

    if instrument_type.upper().startswith("FUT"):
        return {"strikes": [], "source": "futures_no_strikes", "count": 0}

    try:
        expiry_dt = datetime.strptime(expiry_date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(400, "Invalid expiry_date — use DD-MM-YYYY")

    try:
        strikes = await asyncio.to_thread(
            _do_fetch_strikes,
            symbol.strip().upper(),
            instrument_type.upper(),
            expiry_dt,
            option_type.upper(),
        )
    except Exception:
        strikes = []

    source = "historicalOR" if strikes else "not_found"
    return {"strikes": strikes, "source": source, "count": len(strikes)}


@router.get("/api/option-charts/lot-size")
async def oc_lot_size(
    symbol:          str,
    instrument_type: str = "OPTSTK",
    entry_date:      str = "",   
    expiry_date:     str = "",   
):
    symbol = symbol.strip().upper()
    inst   = instrument_type.strip().upper()

    fmt = "%d-%m-%Y"
    entry_dt  = None
    expiry_dt = None
    try:
        if entry_date:
            entry_dt  = datetime.strptime(entry_date.strip(),  fmt).date()
        if expiry_date:
            expiry_dt = datetime.strptime(expiry_date.strip(), fmt).date()
    except ValueError:
        pass   

    try:
        lot = await asyncio.to_thread(_do_fetch_lot_size, symbol, inst, entry_dt, expiry_dt)
        return {"symbol": symbol, "lot_size": lot, "source": "nse"}
    except Exception as e:
        lot = _STATIC_LOT_MAP.get(symbol)
        if lot:
            return {"symbol": symbol, "lot_size": lot, "source": "static"}
        return {"symbol": symbol, "lot_size": None, "source": "unavailable", "error": str(e)}


_STATIC_LOT_MAP: dict[str, int] = {
    "NIFTY": 75, "BANKNIFTY": 30, "FINNIFTY": 40, "MIDCPNIFTY": 75,
    "SENSEX": 10, "BANKEX": 15,
    "RELIANCE": 250, "TCS": 175, "INFY": 300, "HDFCBANK": 550,
    "ICICIBANK": 700, "SBIN": 1500, "WIPRO": 1500, "AXISBANK": 625,
    "KOTAKBANK": 400, "LT": 175, "HINDUNILVR": 300, "ITC": 3200,
    "BAJFINANCE": 125, "MARUTI": 25, "TITAN": 175, "ULTRACEMCO": 100,
    "SUNPHARMA": 700, "HCLTECH": 700, "TATAMOTORS": 1425, "ADANIENT": 250,
    "ADANIPORTS": 1250, "POWERGRID": 4700, "NTPC": 4500, "ONGC": 3850,
    "COALINDIA": 4200, "BPCL": 3500, "TECHM": 600, "DIVISLAB": 150,
    "DRREDDY": 125, "CIPLA": 650, "EICHERMOT": 175, "HEROMOTOCO": 300,
    "BAJAJFINSV": 125, "ASIANPAINT": 200, "NESTLEIND": 50, "BRITANNIA": 100,
    "DABUR": 2500, "PIDILITIND": 400, "GODREJCP": 500, "MARICO": 1200,
    "COLPAL": 350, "BERGEPAINT": 1100, "HAVELLS": 500, "VOLTAS": 500,
    "TATAPOWER": 3375, "TATACONSUM": 900, "TATASTEEL": 5500,
    "HINDALCO": 2150, "JSWSTEEL": 1350, "VEDL": 2000, "SAIL": 6800,
    "GRASIM": 475, "SHREECEM": 25, "AMBUJACEMENT": 2000, "ACC": 500,
    "INDUSINDBK": 500, "FEDERALBNK": 5000, "IDFCFIRSTB": 5000,
    "PNB": 8000, "CANBK": 2300, "BANKBARODA": 5850,
    "ZOMATO": 4500, "NYKAA": 1500, "PAYTM": 2000, "POLICYBZR": 1100,
    "BANDHANBNK": 1800, "RBLBANK": 5000, "IDBI": 7000, "YESBANK": 40000,
    "AUBANK": 500, "DCBBANK": 3000, "SOUTHBANK": 15000, "UJJIVANSFB": 4000,
    "EQUITASBNK": 5000, "CSBBANK": 2000, "IDFC": 8000,
    "ABCAPITAL": 2800, "SBICARD": 500, "SBILIFE": 750, "HDFCLIFE": 1100,
    "ICICIPRULI": 1500, "ICICIGI": 500, "LICI": 700, "STARHEALTH": 500,
    "APOLLOHOSP": 250, "MAXHEALTH": 800, "FORTIS": 2000,
    "AUROPHARMA": 650, "LUPIN": 650, "BIOCON": 2600, "ALKEM": 150,
    "TORNTPHARM": 150, "GLAND": 375, "ABBOTINDIA": 50,
    "MPHASIS": 250, "PERSISTENT": 250, "COFORGE": 200, "LTIM": 150,
    "OFSS": 100, "KPITTECH": 1000,
    "PIIND": 200, "DEEPAKNTR": 250, "TATACHEM": 500, "AARTIIND": 1000,
    "APOLLOTYRE": 2600, "MRF": 10, "BALKRISHNA": 400, "CEAT": 400,
    "GODREJPROP": 350, "DLF": 1500, "OBEROIRLTY": 500, "PRESTIGE": 800,
    "DMART": 350, "TRENT": 350, "INDIGO": 350, "CONCOR": 1000,
    "CUMMINSIND": 600, "BHARATFORG": 1000, "THERMAX": 350,
    "DELHIVERY": 2475, "IRCTC": 875, "IRFC": 7600, "RVNL": 3600,
    "HAL": 150, "BEL": 3700, "BHEL": 4500,
    "MUTHOOTFIN": 375, "CHOLAFIN": 625,
    "JUBLFOOD": 1250, "NAUKRI": 150, "INDHOTEL": 2400,
}


def _do_fetch_lot_size(
    symbol: str,
    instrument_type: str,
    entry_dt=None,   
    expiry_dt=None,  
) -> int:
    from datetime import date, timedelta

    session  = _get_oc_session()
    _warm_up(session)
    api_hdrs = {
        "Accept":           "application/json, text/plain, */*",
        "Referer":          "https://www.nseindia.com/market-data/equity-derivatives-watch",
        "X-Requested-With": "XMLHttpRequest",
    }

    today = date.today()

    def _try_fetch(from_d, to_d, exp_d):
        params = {
            "from":           from_d.strftime("%d-%m-%Y"),
            "to":             to_d.strftime("%d-%m-%Y"),
            "instrumentType": instrument_type,
            "symbol":         symbol,
            "year":           str(exp_d.year),
            "expiryDate":     exp_d.strftime("%d-%b-%Y").upper(),
            "optionType":     "CE",
        }
        try:
            r = session.get(NSE_OC_URL, params=params, headers=api_hdrs, timeout=15)
            if r.status_code != 200:
                return None
            for row in r.json().get("data", []):
                lot = row.get("FH_MARKET_LOT")
                if lot:
                    return int(float(lot))
        except Exception:
            pass
        return None

    if entry_dt and expiry_dt:
        lot = _try_fetch(entry_dt, expiry_dt, expiry_dt)
        if lot:
            return lot
        month_start = date(expiry_dt.year, expiry_dt.month, 1)
        lot = _try_fetch(month_start, expiry_dt, expiry_dt)
        if lot:
            return lot

    for months_ahead in [0, 1, 2, 3]:
        target            = today + timedelta(days=30 * months_ahead)
        expiry_candidates = _get_monthly_expiries(target.year, target.month)
        for exp_candidate in reversed(expiry_candidates):
            if months_ahead == 0 and exp_candidate < today:
                continue
            month_start = date(exp_candidate.year, exp_candidate.month, 1)
            lot = _try_fetch(month_start, exp_candidate, exp_candidate)
            if lot:
                return lot

    raise ValueError(f"Could not fetch lot size for {symbol} from NSE")


def _get_monthly_expiries(year: int, month: int):
    from datetime import date, timedelta
    thursdays = []
    d = date(year, month, 1)
    while d.month == month:
        if d.weekday() == 3:
            thursdays.append(d)
        d += timedelta(days=1)
    return thursdays


@router.post("/api/option-charts/fetch")
async def oc_fetch(req: OcFetchRequest):
    fmt = "%d-%m-%Y"
    try:
        from_dt   = datetime.strptime(req.from_date,   fmt)
        to_dt     = datetime.strptime(req.to_date,     fmt)
        expiry_dt = datetime.strptime(req.expiry_date, fmt)
    except ValueError as exc:
        raise HTTPException(400, f"Date format error (use DD-MM-YYYY): {exc}")

    if from_dt >= to_dt:
        raise HTTPException(400, "from_date must be before to_date.")
    if to_dt.date() > expiry_dt.date():
        raise HTTPException(400, "to_date cannot be after expiry_date.")
    if req.strike_price <= 0:
        raise HTTPException(400, "strike_price must be > 0.")

    try:
        rows = await asyncio.to_thread(
            _do_fetch,
            from_dt, to_dt,
            req.symbol.strip().upper(),
            expiry_dt.year,
            expiry_dt,
            req.option_type.upper(),
            req.strike_price,
            req.instrument_type,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"Internal error: {exc}")

    if not rows:
        raise HTTPException(404, "No rows returned for the given parameters.")

    last = rows[-1]
    return {
        "symbol":      req.symbol.strip().upper(),
        "expiry":      expiry_dt.strftime("%d-%b-%Y"),
        "strike":      req.strike_price,
        "option_type": req.option_type.upper(),
        "rows":        len(rows),
        "last_close":  last.get("close"),
        "last_ltp":    last.get("ltp"),
        "data":        rows,
    }


# 🆕 NEW ENDPOINT: Dedicated Underlying Price API
@router.post("/api/option-charts/underlying-price")
async def get_underlying_price(req: UnderlyingPriceRequest):
    """
    Fetch underlying equity price for a given symbol and date.
    
    Uses the new NSE generateSecurityWiseHistoricalData API as primary source,
    with fallback to options data extraction.
    
    Parameters
    ----------
    symbol          : NSE ticker (e.g., HDFCBANK, RELIANCE, NIFTY)
    target_date     : Date for which you want the price (DD-MM-YYYY)
    instrument_type : OPTSTK/OPTIDX (used only for fallback)
    expiry_date     : Expiry date (used only for fallback)
    
    Returns
    -------
    JSON with:
    - symbol, target_date
    - price: Closing price on/near target_date
    - date: Actual date of the price
    - open, high, low, volume: Additional OHLCV data (if available)
    - source: "equity_api" or "options_fallback"
    """
    if not req.symbol or not req.target_date:
        raise HTTPException(400, "symbol and target_date are required")
    
    try:
        target_dt = datetime.strptime(req.target_date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(400, "Invalid target_date format — use DD-MM-YYYY")
    
    expiry_dt = None
    if req.expiry_date:
        try:
            expiry_dt = datetime.strptime(req.expiry_date, "%d-%m-%Y")
        except ValueError:
            pass
    
    try:
        result = await asyncio.to_thread(
            _get_underlying_price,
            req.symbol.strip().upper(),
            target_dt,
            req.instrument_type.upper(),
            expiry_dt,
            "CE",
        )
        
        if result and result.get("price"):
            return {
                "symbol":      req.symbol.strip().upper(),
                "target_date": req.target_date,
                "success":     True,
                **result,
            }
        else:
            return {
                "symbol":      req.symbol.strip().upper(),
                "target_date": req.target_date,
                "success":     False,
                "error":       f"No price data found for {req.symbol} near {req.target_date}",
                "price":       None,
            }
            
    except Exception as exc:
        raise HTTPException(500, f"Error fetching underlying price: {exc}")


# 🆕 NEW ENDPOINT: GET underlying price (GET version for convenience)
@router.get("/api/option-charts/underlying-price")
async def get_underlying_price_get(
    symbol:      str,
    target_date: str,
):
    """Convenience GET endpoint for underlying price lookup."""
    req = UnderlyingPriceRequest(symbol=symbol, target_date=target_date)
    return await get_underlying_price(req)


def _do_fetch_strikes_with_underlying(
    symbol: str,
    instrument_type: str,
    expiry_dt: datetime,
    option_type: str,
    entry_dt: datetime,
) -> dict:
    """
    🆕 UPDATED: Fetch available CE strikes AND underlying price using new API.
    Primary source for underlying: generateSecurityWiseHistoricalData
    Fallback: FH_UNDERLYING_VALUE from options data
    """
    session = _get_oc_session()
    _warm_up(session)

    # To get strikes: use narrow window around expiry
    today = datetime.now().date()
    if expiry_dt.date() >= today:
        to_dt   = datetime.now()
        from_dt = to_dt - timedelta(days=7)
    else:
        to_dt   = expiry_dt
        from_dt = expiry_dt - timedelta(days=3)

    params = {
        "from":           from_dt.strftime("%d-%m-%Y"),
        "to":             to_dt.strftime("%d-%m-%Y"),
        "instrumentType": instrument_type,
        "symbol":         symbol,
        "year":           str(expiry_dt.year),
        "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
        "optionType":     option_type,
    }
    api_hdrs = {
        "Accept":           "application/json, text/plain, */*",
        "Referer":          "https://www.nseindia.com/market-data/equity-derivatives-watch",
        "X-Requested-With": "XMLHttpRequest",
    }

    strikes = set()
    underlying_price = None
    underlying_date  = None
    underlying_source = None

    # ── Pass 1: fetch strikes list ──
    for attempt in range(2):
        try:
            r = session.get(NSE_OC_URL, params=params, headers=api_hdrs, timeout=15)
            if r.status_code == 401:
                _warm_up(session)
                time.sleep(2)
                continue
            if r.status_code not in (200,):
                break
            data = r.json()
            if not data or "data" not in data or not data["data"]:
                break
            for row in data["data"]:
                sv = row.get("FH_STRIKE_PRICE")
                if sv is not None:
                    try:
                        s = int(float(sv))
                        if s > 0:
                            strikes.add(s)
                    except (ValueError, TypeError):
                        pass
            break
        except requests.RequestException:
            if attempt == 1:
                break
            time.sleep(2)
        except Exception:
            break

    # ── Pass 2: 🆕 Use NEW Equity API for underlying price (PRIMARY) ──
    ul_result = _get_underlying_price(symbol, entry_dt, instrument_type, expiry_dt, option_type)
    if ul_result and ul_result.get("price"):
        underlying_price  = ul_result["price"]
        underlying_date   = ul_result.get("date")
        underlying_source = ul_result.get("source", "unknown")

    # ── Pass 3: Fallback to options data if equity API failed ──
    if underlying_price is None and strikes:
        try:
            any_strike  = sorted(strikes)[len(strikes) // 2]
            entry_from  = entry_dt - timedelta(days=5)
            entry_to    = entry_dt + timedelta(days=10)
            if entry_to.date() > expiry_dt.date():
                entry_to = expiry_dt
            entry_params = {
                "from":           entry_from.strftime("%d-%m-%Y"),
                "to":             entry_to.strftime("%d-%m-%Y"),
                "instrumentType": instrument_type,
                "symbol":         symbol,
                "year":           str(expiry_dt.year),
                "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
                "optionType":     option_type,
                "strikePrice":    str(any_strike),
            }
            r2 = session.get(NSE_OC_URL, params=entry_params, headers=api_hdrs, timeout=15)
            if r2.status_code == 200:
                d2 = r2.json()
                if d2 and "data" in d2 and d2["data"]:
                    best_diff = None
                    for row in d2["data"]:
                        uv = row.get("FH_UNDERLYING_VALUE")
                        ts = row.get("FH_TIMESTAMP")
                        if uv and ts:
                            try:
                                uval   = float(uv)
                                row_dt = datetime.strptime(ts, "%d-%b-%Y")
                                diff   = abs((row_dt.date() - entry_dt.date()).days)
                                if best_diff is None or diff < best_diff:
                                    best_diff        = diff
                                    underlying_price = uval
                                    underlying_date  = row_dt.strftime("%d-%m-%Y")
                                    underlying_source = "options_fallback"
                            except Exception:
                                pass
        except Exception:
            pass

    # ── Pass 4: Try each available strike until one returns data ──
    if underlying_price is None and strikes:
        for try_strike in sorted(strikes)[:5]:
            try:
                entry_from = entry_dt - timedelta(days=5)
                entry_to   = min(entry_dt + timedelta(days=10), expiry_dt)
                p3 = {
                    "from":           entry_from.strftime("%d-%m-%Y"),
                    "to":             entry_to.strftime("%d-%m-%Y"),
                    "instrumentType": instrument_type,
                    "symbol":         symbol,
                    "year":           str(expiry_dt.year),
                    "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
                    "optionType":     option_type,
                    "strikePrice":    str(try_strike),
                }
                r3 = session.get(NSE_OC_URL, params=p3, headers=api_hdrs, timeout=15)
                if r3.status_code == 200:
                    d3 = r3.json()
                    if d3 and "data" in d3 and d3["data"]:
                        best_diff = None
                        for row in d3["data"]:
                            uv = row.get("FH_UNDERLYING_VALUE")
                            ts = row.get("FH_TIMESTAMP")
                            if uv and ts:
                                try:
                                    uval   = float(uv)
                                    row_dt = datetime.strptime(ts, "%d-%b-%Y")
                                    diff   = abs((row_dt.date() - entry_dt.date()).days)
                                    if best_diff is None or diff < best_diff:
                                        best_diff        = diff
                                        underlying_price = uval
                                        underlying_date  = row_dt.strftime("%d-%m-%Y")
                                        underlying_source = "options_fallback_multi_strike"
                                except Exception:
                                    pass
                        if underlying_price is not None:
                            break
            except Exception:
                continue

    strikes_sorted = sorted(strikes)

    # ── Pass 1b: Fetch entry-day OHLC for premium + chg_pct + frm_low ──
    # Two bulk calls:
    #   Call A: entry_dt-10d → entry_dt   → gets entry-day close + any prev-day rows in the 70-row window
    #   Call B: entry_dt-5d  → entry_dt-1d → dedicated prev-day bulk fetch to fill gaps
    # If both bulk calls return 0 rows (session not yet warmed), retry up to 2x with re-warm.
    from collections import defaultdict

    strike_ohlc_map: dict = {}   # strike -> {"entry_close", "prev_close", "period_low"}

    def _parse_bulk_rows(raw_rows: list) -> "defaultdict[int, list]":
        """Parse raw NSE foCPV rows into {strike_int: [{dt, close, low}]} map."""
        smap: defaultdict = defaultdict(list)
        for row in raw_rows:
            sv = row.get("FH_STRIKE_PRICE")
            ts = row.get("FH_TIMESTAMP")
            cl = row.get("FH_CLOSING_PRICE") or row.get("FH_LAST_TRADED_PRICE")
            lo = row.get("FH_TRADE_LOW_PRICE")
            if sv is None or ts is None or cl is None:
                continue
            try:
                s_int  = int(float(sv))
                row_dt = datetime.strptime(str(ts).strip(), "%d-%b-%Y")
                close  = float(cl)
                low    = float(lo) if lo else close
                smap[s_int].append({"dt": row_dt, "close": close, "low": low})
            except Exception:
                continue
        return smap

    def _bulk_fetch(from_str: str, to_str: str) -> list:
        """Single bulk fetch attempt; returns raw rows list (may be empty)."""
        params_b = {
            "from":           from_str,
            "to":             to_str,
            "instrumentType": instrument_type,
            "symbol":         symbol,
            "year":           str(expiry_dt.year),
            "expiryDate":     expiry_dt.strftime("%d-%b-%Y").upper(),
            "optionType":     option_type,
        }
        r_b = session.get(NSE_OC_URL, params=params_b, headers=api_hdrs, timeout=15)
        if r_b.status_code == 401:
            _warm_up(session)
            time.sleep(2)
            r_b = session.get(NSE_OC_URL, params=params_b, headers=api_hdrs, timeout=15)
        if r_b.status_code != 200:
            return []
        d_b = r_b.json()
        return d_b.get("data", []) if d_b else []

    try:
        entry_date_only = entry_dt.date()
        p1b_from = (entry_dt - timedelta(days=10)).strftime("%d-%m-%Y")
        p1b_to   = entry_dt.strftime("%d-%m-%Y")

        # ── Call A: entry-day window (retry up to 3x if empty — session warming) ──
        rows_a = []
        for _attempt in range(3):
            print(f"[PASS1B] Call-A attempt {_attempt+1}: {p1b_from} → {p1b_to}")
            rows_a = _bulk_fetch(p1b_from, p1b_to)
            print(f"[PASS1B] Call-A rows={len(rows_a)}")
            if rows_a:
                break
            if _attempt < 2:
                _warm_up(session)
                time.sleep(2)

        strike_rows_a = _parse_bulk_rows(rows_a)
        dates_a = sorted({r["dt"].strftime("%d-%b-%Y") for rows in strike_rows_a.values() for r in rows})
        print(f"[PASS1B] Call-A dates: {dates_a}  strikes: {len(strike_rows_a)}")

        # ── Call B: prev-day window — dedicated fetch for prior close ──
        prev_from = (entry_dt - timedelta(days=5)).strftime("%d-%m-%Y")
        prev_to   = (entry_dt - timedelta(days=1)).strftime("%d-%m-%Y")
        print(f"[PASS1B] Call-B: {prev_from} → {prev_to}")
        rows_b = _bulk_fetch(prev_from, prev_to)
        strike_rows_b = _parse_bulk_rows(rows_b)
        dates_b = sorted({r["dt"].strftime("%d-%b-%Y") for rows in strike_rows_b.values() for r in rows})
        print(f"[PASS1B] Call-B rows={len(rows_b)}  dates: {dates_b}  strikes: {len(strike_rows_b)}")

        # Merge: build ohlc_map from both calls
        all_strikes_seen = set(strike_rows_a.keys()) | set(strike_rows_b.keys())
        prev_ok = prev_missing = 0
        for s_int in all_strikes_seen:
            rows_a_s = sorted(strike_rows_a.get(s_int, []), key=lambda x: x["dt"])
            rows_b_s = sorted(strike_rows_b.get(s_int, []), key=lambda x: x["dt"])

            # Entry close: must be on entry_dt (from Call A)
            entry_row = next((r for r in rows_a_s if r["dt"].date() == entry_date_only), None)
            if not entry_row:
                continue  # No entry-date data for this strike — skip

            # Prev close: check Call A first (some strikes have 2 rows), then Call B
            before_a = [r for r in rows_a_s if r["dt"].date() < entry_date_only]
            before_b = [r for r in rows_b_s if r["dt"].date() < entry_date_only]
            all_before = sorted(before_a + before_b, key=lambda x: x["dt"])
            prev_row = all_before[-1] if all_before else None

            # Period low: all rows up to entry_dt across both calls
            all_rows = sorted(rows_a_s + rows_b_s, key=lambda x: x["dt"])
            period_low = min(
                (r["low"] for r in all_rows if r["dt"].date() <= entry_date_only),
                default=None
            )

            strike_ohlc_map[s_int] = {
                "entry_close": entry_row["close"],
                "prev_close":  prev_row["close"] if prev_row else None,
                "period_low":  period_low,
            }
            if prev_row:
                prev_ok += 1
            else:
                prev_missing += 1

        print(f"[PASS1B] map built: {len(strike_ohlc_map)} strikes  prev_ok={prev_ok}  prev_missing={prev_missing}")

        # ── FALLBACK: both bulk calls returned 0 entry-date rows ──
        # NSE session may be cold for this symbol; try per-strike fetch for ATM±3 strikes only.
        if len(strike_ohlc_map) == 0 and strikes:
            print(f"[PASS1B] bulk missed entry date — per-strike fallback")
            mid = len(strikes) // 2
            probe_strikes = sorted(strikes)[max(0, mid - 3):mid + 4]
            confirmed_window = False
            for probe_s in probe_strikes:
                try:
                    ps_params = {
                        "from": p1b_from, "to": p1b_to,
                        "instrumentType": instrument_type, "symbol": symbol,
                        "year": str(expiry_dt.year),
                        "expiryDate": expiry_dt.strftime("%d-%b-%Y").upper(),
                        "optionType": option_type, "strikePrice": str(probe_s),
                    }
                    r_ps = session.get(NSE_OC_URL, params=ps_params, headers=api_hdrs, timeout=15)
                    if r_ps.status_code != 200:
                        continue
                    d_ps = r_ps.json()
                    probe_rows = _parse_bulk_rows(d_ps.get("data", []) if d_ps else []).get(probe_s, [])
                    has_entry = any(r["dt"].date() == entry_date_only for r in probe_rows)
                    print(f"[PASS1B] probe strike={probe_s} rows={len(probe_rows)} has_entry={has_entry}")
                    if has_entry:
                        confirmed_window = True
                        break
                except Exception as _pe:
                    print(f"[PASS1B] probe error: {_pe}")

            if confirmed_window:
                # Full per-strike sweep
                for tgt_s in sorted(strikes):
                    try:
                        ps_a = {
                            "from": p1b_from, "to": p1b_to,
                            "instrumentType": instrument_type, "symbol": symbol,
                            "year": str(expiry_dt.year),
                            "expiryDate": expiry_dt.strftime("%d-%b-%Y").upper(),
                            "optionType": option_type, "strikePrice": str(tgt_s),
                        }
                        r_a = session.get(NSE_OC_URL, params=ps_a, headers=api_hdrs, timeout=10)
                        rows_ta = _parse_bulk_rows(r_a.json().get("data", []) if r_a.status_code == 200 else []).get(tgt_s, [])
                        ps_b = {**ps_a, "from": prev_from, "to": prev_to}
                        r_b2 = session.get(NSE_OC_URL, params=ps_b, headers=api_hdrs, timeout=10)
                        rows_tb = _parse_bulk_rows(r_b2.json().get("data", []) if r_b2.status_code == 200 else []).get(tgt_s, [])
                        e_row = next((r for r in sorted(rows_ta, key=lambda x: x["dt"]) if r["dt"].date() == entry_date_only), None)
                        if not e_row:
                            continue
                        all_bef = sorted([r for r in rows_ta + rows_tb if r["dt"].date() < entry_date_only], key=lambda x: x["dt"])
                        p_row = all_bef[-1] if all_bef else None
                        p_low = min((r["low"] for r in rows_ta + rows_tb if r["dt"].date() <= entry_date_only), default=None)
                        strike_ohlc_map[tgt_s] = {
                            "entry_close": e_row["close"],
                            "prev_close":  p_row["close"] if p_row else None,
                            "period_low":  p_low,
                        }
                        time.sleep(0.15)
                    except Exception:
                        continue
                print(f"[PASS1B] per-strike sweep complete: {len(strike_ohlc_map)} strikes mapped")

    except Exception as _e1b:
        print(f"[PASS1B] exception: {_e1b}")

    # Build strike_detail with OTM/ATM/ITM classification + chg_pct + frm_low
    strikes_detail = []
    for s in strikes_sorted:
        if underlying_price is not None:
            pct_otm = round((s - underlying_price) / underlying_price * 100, 1)
            if abs(pct_otm) <= 0.5:
                zone = "ATM"
            elif pct_otm > 0:
                zone = "OTM"
            else:
                zone = "ITM"
        else:
            pct_otm = None
            zone    = "UNK"
        # chg_pct and frm_low from Pass 1b
        ohlc = strike_ohlc_map.get(s)
        chg_pct = None
        frm_low = None
        entry_close = None
        if ohlc:
            entry_close = ohlc["entry_close"]
            if ohlc["prev_close"] and ohlc["prev_close"] > 0:
                chg_pct = round((entry_close - ohlc["prev_close"]) / ohlc["prev_close"] * 100, 1)
            if ohlc["period_low"] and ohlc["period_low"] > 0 and entry_close:
                frm_low = round((entry_close - ohlc["period_low"]) / ohlc["period_low"] * 100, 1)
        strikes_detail.append({
            "strike":       s,
            "pct_otm":      pct_otm,
            "zone":         zone,
            "premium":      entry_close,
            "chg_pct":      chg_pct,
            "frm_low":      frm_low,
        })

    # Build suggested spread pairs
    spread_pairs = []
    if underlying_price is not None:
        otm_strikes = [s for s in strikes_sorted if s > underlying_price]
        step = None
        if len(otm_strikes) >= 2:
            diffs = [otm_strikes[i+1] - otm_strikes[i] for i in range(min(5, len(otm_strikes)-1))]
            if diffs:
                step = int(sorted(diffs)[len(diffs)//2])

        for i, short_s in enumerate(otm_strikes[:12]):
            for width_mult in [1, 2, 3]:
                long_s = short_s + (step * width_mult if step else 0)
                if long_s in strikes and long_s != short_s:
                    short_pct = round((short_s - underlying_price) / underlying_price * 100, 1)
                    entry = {"short_strike": short_s, "long_strike": long_s,
                             "spread_width": long_s - short_s, "short_pct_otm": short_pct}
                    if entry not in spread_pairs:
                        spread_pairs.append(entry)
                    if len(spread_pairs) >= 12:
                        break
            if len(spread_pairs) >= 12:
                break

    return {
        "strikes":           strikes_sorted,
        "strikes_detail":    strikes_detail,
        "underlying":        round(underlying_price, 2) if underlying_price else None,
        "underlying_date":   underlying_date,
        "underlying_source": underlying_source,
        "spread_pairs":      spread_pairs,
    }


@router.get("/api/option-charts/spread/suggest-strikes")
async def spread_suggest_strikes(
    symbol:          str,
    instrument_type: str = "OPTSTK",
    expiry_date:     str = "",
    entry_date:      str = "",
    option_type:     str = "CE",
    num_otm_levels:  int = 5,
):
    """
    🆕 UPDATED: Now returns underlying_source to show which API provided the price.
    """
    if not symbol or not expiry_date or not entry_date:
        raise HTTPException(400, "symbol, expiry_date and entry_date are required")

    try:
        expiry_dt = datetime.strptime(expiry_date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(400, "Invalid expiry_date — use DD-MM-YYYY")

    try:
        entry_dt = datetime.strptime(entry_date, "%d-%m-%Y")
    except ValueError:
        raise HTTPException(400, "Invalid entry_date — use DD-MM-YYYY")

    try:
        result = await asyncio.to_thread(
            _do_fetch_strikes_with_underlying,
            symbol.strip().upper(),
            instrument_type.upper(),
            expiry_dt,
            option_type.upper(),
            entry_dt,
        )
    except Exception as exc:
        raise HTTPException(500, f"Error fetching strike suggestions: {exc}")

    return result


@router.post("/api/option-charts/spread/bear-call")
async def bear_call_spread(req: BearCallSpreadRequest):
    """
    🆕 UPDATED: Bear Call Spread analysis now uses new equity API for 
    underlying price detection. Response includes 'underlying_source' field.
    """
    fmt = "%d-%m-%Y"

    try:
        entry_dt   = datetime.strptime(req.entry_date, fmt)
        expiry1_dt = datetime.strptime(req.expiry_date_1, fmt)
    except ValueError as exc:
        raise HTTPException(400, f"Date format error (use DD-MM-YYYY): {exc}")

    if req.short_strike <= 0:
        raise HTTPException(400, "short_strike must be > 0")
    if req.long_strike <= 0:
        raise HTTPException(400, "long_strike must be > 0")
    if req.long_strike <= req.short_strike:
        raise HTTPException(400, "long_strike must be > short_strike for a Bear Call Spread")
    if entry_dt.date() > expiry1_dt.date():
        raise HTTPException(400, "entry_date must be before or on expiry_date_1")
    if req.lot_size <= 0:
        raise HTTPException(400, "lot_size must be > 0")
    if req.num_lots <= 0:
        raise HTTPException(400, "num_lots must be > 0")

    symbol    = req.symbol.strip().upper()
    inst_type = req.instrument_type.upper()
    entry_str = entry_dt.strftime("%Y-%m-%d")

    result = {"symbol": symbol, "instrument_type": inst_type}

    # Expiry 1
    try:
        e1 = await asyncio.to_thread(
            _do_bear_call_spread,
            symbol, inst_type,
            req.short_strike, req.long_strike,
            expiry1_dt, entry_str,
            req.lot_size, req.num_lots,
        )
        result["expiry_1"] = e1
    except ValueError as exc:
        raise HTTPException(400, f"Expiry 1 error: {exc}")
    except Exception as exc:
        raise HTTPException(500, f"Expiry 1 internal error: {exc}")

    # Expiry 2 (optional)
    if req.expiry_date_2:
        try:
            expiry2_dt = datetime.strptime(req.expiry_date_2, fmt)
        except ValueError as exc:
            raise HTTPException(400, f"expiry_date_2 format error: {exc}")

        if entry_dt.date() > expiry2_dt.date():
            raise HTTPException(400, "entry_date must be before or on expiry_date_2")

        try:
            e2 = await asyncio.to_thread(
                _do_bear_call_spread,
                symbol, inst_type,
                req.short_strike, req.long_strike,
                expiry2_dt, entry_str,
                req.lot_size, req.num_lots,
            )
            result["expiry_2"] = e2
        except ValueError as exc:
            result["expiry_2_error"] = str(exc)
        except Exception as exc:
            result["expiry_2_error"] = f"Internal error: {exc}"

    return result