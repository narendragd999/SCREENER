"""
Fair Value API Router  —  /api/fv/
===================================
Reusable FastAPI router extracted from sma_router.py.
Provides year-wise and quarter-wise fair value data with:
  • OP Fair Value      (Operating Profit linear regression)
  • Sales Fair Value   (Revenue / Sales linear regression)
  • TTM Fair Value     (Trailing-twelve-month OP & Sales regression)
  • Composite Fair Value (R²-weighted blend of all active models)

Usage (in main.py):
    from fair_value_router import router as fv_router
    app.include_router(fv_router)

Endpoints:
    GET  /api/fv/fair-value/{ticker}
    POST /api/fv/fair-value/batch
    GET  /api/fv/fair-value/{ticker}/yearly
    GET  /api/fv/fair-value/{ticker}/quarterly
    GET  /api/fv/fair-value/{ticker}/summary
    GET  /api/fv/cache/clear?ticker=RELIANCE        (optional filter)
"""

import os
import re
import asyncio
import datetime as dt_module
import difflib
from typing import Optional, List, Dict, Any, Literal

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

router = APIRouter(prefix="/api/fv", tags=["Fair Value"])

# ─────────────────────────────────────────────────────────────
#  CONFIG  (override via env-vars if needed)
# ─────────────────────────────────────────────────────────────
DATA_DIR = os.getenv("FV_DATA_DIR", "data")
FNO_CSV  = os.getenv("FNO_CSV", "tickers.csv")        # Sr. No., SECURITY, SYMBOL
ALL_CSV  = os.getenv("ALL_CSV", "tickers_all.csv")    # SYMBOL, NAME OF COMPANY
os.makedirs(DATA_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────
#  TICKER CACHE
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
#  HELPERS
# ─────────────────────────────────────────────────────────────
_BANK_KEYWORDS = {"BANK", "FINANCE", "NBFC", "FINANCIAL", "LENDING", "MICROFINANCE"}
_KNOWN_BANKS   = {
    "HDFCBANK", "ICICIBANK", "AXISBANK", "KOTAKBANK", "SBIN", "BANDHANBNK",
    "INDUSINDBK", "FEDERALBNK", "RBLBANK", "IDFCFIRSTB", "YESBANK", "PNB",
    "CANBK", "BANKBARODA", "UNIONBANK", "AUBANK", "DCBBANK", "KARNATAKABAN",
}

_SLUG_OVERRIDES = {
    "DMART":     "AVENUE-SUPERMARTS",
    "M&M":       "M-AND-M",
    "M&MFIN":    "M-AND-MFIN",
    "L&TFH":     "L-AND-TFH",
    "BAJAJ-AUTO": "BAJAJ-AUTO",
}


def _is_bank_or_finance(ticker: str) -> bool:
    ticker = ticker.upper()
    df = _load_fno_df()
    if not df.empty:
        row = df[df["symbol"] == ticker]
        if not row.empty:
            name = row["company_name"].iloc[0].upper()
            return any(k in name for k in _BANK_KEYWORDS)
    return ticker in _KNOWN_BANKS


def _screener_slug(ticker: str) -> str:
    if ticker in _SLUG_OVERRIDES:
        return _SLUG_OVERRIDES[ticker]
    return ticker.replace("&", "-AND-").replace(" ", "-")


def _fetch_html(url: str) -> Optional[str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer":         "https://www.screener.in/",
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            return r.text
        print(f"[FV] {url} → HTTP {r.status_code}")
    except Exception as exc:
        print(f"[FV] {url} → {exc}")
    return None


def _parse_screener_table(table) -> Optional[pd.DataFrame]:
    try:
        thead = table.find("thead")
        if not thead:
            return None
        headers = [th.get_text(strip=True) for th in thead.find_all("th")]
        if len(headers) < 2:
            return None
        if not headers[0]:
            headers[0] = "Metric"
        tbody = table.find("tbody")
        if not tbody:
            return None
        rows = []
        for tr in tbody.find_all("tr"):
            cells = [td.get_text(strip=True) for td in tr.find_all("td")]
            if len(cells) == len(headers):
                rows.append(cells)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=headers)
        df.set_index("Metric", inplace=True)
        return df
    except Exception:
        return None


def _scrape_section(ticker: str, section_id: str) -> Optional[pd.DataFrame]:
    slug = _screener_slug(ticker)
    dfs = []
    for url in [
        f"https://www.screener.in/company/{slug}/consolidated/",
        f"https://www.screener.in/company/{slug}/",
    ]:
        html = _fetch_html(url)
        if not html:
            continue
        soup    = BeautifulSoup(html, "html.parser")
        section = soup.find("section", {"id": section_id})
        if not section:
            continue
        table = section.find("table", class_="data-table")
        if not table:
            continue
        df = _parse_screener_table(table)
        if df is not None and not df.empty:
            dfs.append(df)
    if not dfs:
        return None
    if len(dfs) == 1:
        return dfs[0]
    # Pick the one with more non-zero profit data points
    best, best_n = dfs[0], 0
    for d in dfs:
        idx, _ = _find_row(d, "profit")
        if idx:
            s = _extract_annual_series(idx, d)
            n = int((s != 0).sum())
            if n > best_n:
                best, best_n = d, n
    return best


def _clean_numeric(s: pd.Series) -> pd.Series:
    return (
        s.astype(str)
         .str.replace(",", "", regex=False)
         .str.replace(r"[^\d.-]", "", regex=True)
         .str.strip()
         .replace("", "0")
         .pipe(pd.to_numeric, errors="coerce")
         .fillna(0)
    )


_ROW_CANDIDATES = {
    "profit": [
        ("Operating Profit",       "Operating Profit"),
        ("OP",                     "Operating Profit"),
        ("EBIT",                   "Operating Profit"),
        ("EBITDA",                 "Operating Profit"),
        ("Financing Profit",       "Financing Profit"),
        ("Interest Income",        "Interest Income"),
        ("Net Interest Income",    "Net Interest Income"),
    ],
    "sales": [
        ("Sales",           "Sales"),
        ("Revenue",         "Sales"),
        ("Net Sales",       "Sales"),
        ("Total Income",    "Sales"),
        ("Interest Earned", "Interest Earned"),
        ("Income",          "Income"),
    ],
    "other_income": [
        ("Other Income",           "Other Income"),
        ("Other Operating Income", "Other Income"),
    ],
}


def _find_row(df: pd.DataFrame, name: str):
    target = name.lower()
    for keyword, display in _ROW_CANDIDATES.get(target, []):
        for idx in df.index:
            if keyword.lower() in idx.lower():
                return idx, display
    idx_lower = [i.lower().strip() for i in df.index]
    matches = difflib.get_close_matches(target, idx_lower, n=1, cutoff=0.6)
    if matches:
        matched = df.index[idx_lower.index(matches[0])]
        label   = "Operating Profit" if target == "profit" else "Sales"
        return matched, label
    return None, "Unknown"


def _extract_annual_series(row_idx, df: pd.DataFrame) -> pd.Series:
    """Extract year-indexed (int) numeric series from an annual P&L table."""
    if row_idx is None:
        return pd.Series(dtype=float)
    series   = df.loc[row_idx]
    raw_cols = df.columns.tolist()
    years, vals = [], []
    for col in raw_cols:
        col_str = str(col).strip()
        if col_str.upper() == "TTM":
            continue
        m = re.search(r"\d{2,4}", col_str)
        if m:
            yr_str = m.group(0)
            yr = int(yr_str)
            if len(yr_str) == 2:
                yr += 2000
            if 2000 <= yr <= 2100:
                years.append(yr)
                vals.append(series[col])
                continue
        try:
            yr = int(col_str)
            if 2000 <= yr <= 2100:
                years.append(yr)
                vals.append(series[col])
        except Exception:
            continue
    if not years:
        return pd.Series(dtype=float)
    clean = _clean_numeric(pd.Series(vals, index=years))
    return clean[clean != 0].dropna()


def _extract_quarterly_series(row_idx, df: pd.DataFrame) -> pd.Series:
    """Extract quarter-end datetime-indexed numeric series from a quarterly table."""
    if row_idx is None:
        return pd.Series(dtype=float)
    series   = df.loc[row_idx]
    raw_cols = df.columns.tolist()
    _month_num  = {"Mar": 3, "Jun": 6, "Sep": 9, "Dec": 12}
    _month_days = {"Mar": 31, "Jun": 30, "Sep": 30, "Dec": 31}
    q_dates, vals = [], []
    for col in raw_cols:
        col_str = str(col).strip()
        m = re.match(r"([A-Za-z]{3})\s+(\d{4})", col_str)
        if m:
            mon_abbr = m.group(1)
            year     = int(m.group(2))
            if mon_abbr in _month_num:
                mon = _month_num[mon_abbr]
                day = _month_days[mon_abbr]
                try:
                    q_date = dt_module.date(year, mon, day)
                    q_dates.append(pd.to_datetime(q_date))
                    vals.append(series[col])
                except ValueError:
                    continue
    if not q_dates:
        return pd.Series(dtype=float)
    raw = pd.Series(vals, index=q_dates)
    return _clean_numeric(raw).dropna()


# ─────────────────────────────────────────────────────────────
#  DISK CACHE  (per-ticker CSV)
# ─────────────────────────────────────────────────────────────
def _get_cached(ticker: str, kind: str, force: bool) -> Optional[pd.DataFrame]:
    path = os.path.join(DATA_DIR, f"{ticker}_{kind}.csv")
    if not force and os.path.exists(path):
        try:
            return pd.read_csv(path, index_col=0)
        except Exception:
            pass
    return None


def _save_cache(ticker: str, kind: str, df: pd.DataFrame):
    try:
        df.to_csv(os.path.join(DATA_DIR, f"{ticker}_{kind}.csv"))
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────
#  LINEAR REGRESSION HELPER
# ─────────────────────────────────────────────────────────────
def _run_lr(x: np.ndarray, y: np.ndarray):
    """Fit LinearRegression; return (model, r2, intercept, coef)."""
    model = LinearRegression().fit(x.reshape(-1, 1), y)
    r2    = round(model.score(x.reshape(-1, 1), y), 3)
    return model, r2, round(float(model.intercept_), 4), round(float(model.coef_[0]), 6)


def _historical_misprice(model, x_vals, y_vals, labels, x_key: str) -> Dict[str, Any]:
    hist = {}
    for i, lbl in enumerate(labels):
        fair   = round(float(model.predict([[x_vals[i]]])[0]), 2)
        actual = round(float(y_vals[i]), 2)
        mp     = round((actual - fair) / fair * 100, 1) if fair else 0
        hist[str(lbl)] = {
            x_key:          round(float(x_vals[i]), 2),
            "fair_value":   fair,
            "actual_price": actual,
            "misprice_pct": mp,
            "status":       "Overvalued" if mp > 20 else ("Undervalued" if mp < -20 else "Fair"),
        }
    return hist


# ─────────────────────────────────────────────────────────────
#  COMPOSITE SCORE
# ─────────────────────────────────────────────────────────────
def _compute_composite(op: Dict, sales: Dict, ttm: Dict) -> Dict:
    """
    R²-weighted blend of OP, Sales, TTM fair value models.
    TTM gets 1.2× weight; Sales gets 0.8× weight.
    """
    gains, prices, weights = [], [], []
    for block, w_mult in [(op, 1.0), (sales, 0.8), (ttm, 1.2)]:
        if block and block.get("pred_price") and float(block.get("r2", 0)) > 0.1:
            w = max(0.1, float(block["r2"])) * w_mult
            gains.append(float(block["gain_pct"]))
            prices.append(float(block["pred_price"]))
            weights.append(w)
    if not gains:
        return {
            "composite_fair_price": None,
            "composite_gain_pct":   None,
            "valuation_bucket":     "INSUFFICIENT_DATA",
            "model_count":          0,
        }
    total_w        = sum(weights)
    composite_gain = sum(g * w for g, w in zip(gains, weights)) / total_w
    composite_fair = sum(p * w for p, w in zip(prices, weights)) / total_w
    return {
        "composite_fair_price": round(composite_fair, 2),
        "composite_gain_pct":   round(composite_gain, 2),
        "valuation_bucket":     (
            "UNDERVALUED" if composite_gain > 15
            else ("OVERVALUED" if composite_gain < -15 else "FAIR")
        ),
        "model_count": len(gains),
    }


# ─────────────────────────────────────────────────────────────
#  CORE COMPUTATION
# ─────────────────────────────────────────────────────────────
def _compute_fair_value(
    ticker:               str,
    fy_start:             int  = 2014,
    force_scrape:         bool = False,
    include_other_income: bool = True,
) -> Dict:
    """
    Full fair-value computation for one ticker.
    Returns a rich dict with yearly, quarterly (TTM), and composite data.
    Raises ValueError on unrecoverable errors.
    """
    ticker  = ticker.strip().upper().lstrip("$").strip()
    is_bank = _is_bank_or_finance(ticker)

    # ── P&L (annual) ──────────────────────────────────────────
    pl_df = _get_cached(ticker, "pl", force_scrape)
    if pl_df is None:
        pl_df = _scrape_section(ticker, "profit-loss")
        if pl_df is not None:
            _save_cache(ticker, "pl", pl_df)
    if pl_df is None or pl_df.empty:
        raise ValueError("Could not scrape P&L from Screener.in")

    profit_row, profit_label = _find_row(pl_df, "profit")
    if not profit_row:
        raise ValueError("No Operating Profit row found in P&L table")

    sales_row, sales_label = _find_row(pl_df, "sales")
    if is_bank and sales_row is None and len(pl_df) > 0:
        fi = pl_df.index[0]
        if any(w in fi.lower() for w in ["income", "total", "revenue"]):
            sales_row, sales_label = fi, "Total Income"

    profit_s = _extract_annual_series(profit_row, pl_df)
    sales_s  = _extract_annual_series(sales_row, pl_df) if sales_row else pd.Series(dtype=float)

    # Optionally add other income for banks
    if include_other_income and is_bank:
        oi_row, _ = _find_row(pl_df, "other_income")
        if oi_row:
            oi_s    = _extract_annual_series(oi_row, pl_df)
            common  = profit_s.index.intersection(oi_s.index)
            if len(common) >= 2:
                profit_s     = profit_s.copy()
                profit_s.loc[common] += oi_s.loc[common]
                profit_label = f"{profit_label} + Other Income"

    if profit_s.empty:
        raise ValueError("No numeric Profit data found")

    # ── Price history (yfinance) ───────────────────────────────
    try:
        t        = yf.Ticker(f"{ticker}.NS")
        price_df = t.history(
            start=f"{fy_start}-04-01",
            end=f"{dt_module.date.today().year + 1}-03-31",
            auto_adjust=True,
        )
        if price_df is not None and not price_df.empty:
            price_df.index = (
                price_df.index.tz_localize(None)
                if price_df.index.tz is None
                else price_df.index.tz_convert(None)
            )
    except Exception as exc:
        raise ValueError(f"yfinance error: {exc}")

    if price_df is None or price_df.empty:
        raise ValueError("No price data from yfinance")

    # FY-end prices (April–March fiscal year)
    price_df["FY_Year"] = price_df.index.year
    price_df.loc[price_df.index.month <= 3, "FY_Year"] -= 1
    fy_end_price  = price_df.groupby("FY_Year")["Close"].last().round(2)
    current_price = round(float(price_df["Close"].iloc[-1]), 2)

    # ── Annual OP model ───────────────────────────────────────
    common_yrs = sorted(profit_s.index.intersection(fy_end_price.index))
    if len(common_yrs) < 2:
        raise ValueError("Need ≥2 overlapping years of OP + price data")

    op_x  = np.array(profit_s.loc[common_yrs]).flatten()
    op_y  = np.array(fy_end_price.loc[common_yrs]).flatten()
    m_op, r2_op, b_op, c_op = _run_lr(op_x, op_y)
    pred_op  = round(float(m_op.predict([[op_x[-1]]])[0]), 2)
    gain_op  = round((pred_op - current_price) / current_price * 100, 2) if current_price else 0

    yearly_op = {
        "model": "OP",
        "label": profit_label,
        "equation":   f"Price = {b_op} + {c_op} × {profit_label.split()[0]}",
        "r2":          r2_op,
        "pred_price":  pred_op,
        "gain_pct":    gain_op,
        "data": [
            {
                "year":         int(y),
                "op":           round(float(op_x[i]), 2),
                "price":        round(float(op_y[i]), 2),
                "price_to_op":  round(float(op_y[i] / op_x[i]), 4) if op_x[i] else None,
                "fair_value":   round(float(m_op.predict([[op_x[i]]])[0]), 2),
                "misprice_pct": round(
                    (float(op_y[i]) - float(m_op.predict([[op_x[i]]])[0]))
                    / float(m_op.predict([[op_x[i]]])[0]) * 100, 1
                ) if float(m_op.predict([[op_x[i]]])[0]) else 0,
            }
            for i, y in enumerate(common_yrs)
        ],
        "historical": _historical_misprice(m_op, op_x, op_y, common_yrs, "op"),
    }

    # ── Annual Sales model ────────────────────────────────────
    yearly_sales: Dict = {}
    has_sales = False
    if not sales_s.empty:
        sy = sorted(sales_s.index.intersection(fy_end_price.index))
        if len(sy) >= 2:
            has_sales = True
            sv = np.array(sales_s.loc[sy]).flatten()
            spy = np.array(fy_end_price.loc[sy]).flatten()
            m_s, r2_s, b_s, c_s = _run_lr(sv, spy)
            pred_s  = round(float(m_s.predict([[sv[-1]]])[0]), 2)
            gain_s  = round((pred_s - current_price) / current_price * 100, 2) if current_price else 0
            yearly_sales = {
                "model":    "Sales",
                "label":    sales_label,
                "equation": f"Price = {b_s} + {c_s} × {sales_label.split()[0]}",
                "r2":        r2_s,
                "pred_price": pred_s,
                "gain_pct":  gain_s,
                "data": [
                    {
                        "year":          int(y),
                        "sales":         round(float(sv[i]), 2),
                        "price":         round(float(spy[i]), 2),
                        "price_to_sales": round(float(spy[i] / sv[i]), 4) if sv[i] else None,
                        "fair_value":    round(float(m_s.predict([[sv[i]]])[0]), 2),
                        "misprice_pct":  round(
                            (float(spy[i]) - float(m_s.predict([[sv[i]]])[0]))
                            / float(m_s.predict([[sv[i]]])[0]) * 100, 1
                        ) if float(m_s.predict([[sv[i]]])[0]) else 0,
                    }
                    for i, y in enumerate(sy)
                ],
                "historical": _historical_misprice(m_s, sv, spy, sy, "sales"),
            }

    # ── Quarterly / TTM models ────────────────────────────────
    ttm_op:    Dict = {}
    ttm_sales: Dict = {}
    has_ttm_op    = False
    has_ttm_sales = False

    qr_df = _get_cached(ticker, "qr", force_scrape)
    if qr_df is None:
        qr_df = _scrape_section(ticker, "quarters")
        if qr_df is not None:
            _save_cache(ticker, "qr", qr_df)

    if qr_df is not None and not qr_df.empty:
        p_row_q, _        = _find_row(qr_df, "profit")
        s_row_q, s_lbl_q  = _find_row(qr_df, "sales")
        if is_bank and s_row_q is None and len(qr_df) > 0:
            fi = qr_df.index[0]
            if any(w in fi.lower() for w in ["income", "total", "revenue"]):
                s_row_q, s_lbl_q = fi, "Total Income"

        price_qe = price_df["Close"].resample("QE").last().round(2)

        # TTM OP
        if p_row_q:
            profit_q = _extract_quarterly_series(p_row_q, qr_df).sort_index()
            if include_other_income and is_bank:
                oi_qr, _ = _find_row(qr_df, "other_income")
                if oi_qr:
                    oi_q = _extract_quarterly_series(oi_qr, qr_df)
                    cq   = profit_q.index.intersection(oi_q.index)
                    if len(cq):
                        profit_q = profit_q.copy()
                        profit_q.loc[cq] += oi_q.loc[cq]
            if len(profit_q) >= 4:
                ttm_vals = [profit_q.iloc[i - 3:i + 1].sum() for i in range(3, len(profit_q))]
                ttm_idx  = profit_q.index[3:]
                ttm_s_op = pd.Series(ttm_vals, index=ttm_idx)
                common_t = sorted(ttm_s_op.index.intersection(price_qe.index))
                if len(common_t) >= 2:
                    has_ttm_op = True
                    tv  = np.array(ttm_s_op.loc[common_t]).flatten()
                    pqv = np.array(price_qe.loc[common_t]).flatten()
                    m_t, r2_t, b_t, c_t = _run_lr(tv, pqv)
                    pred_t = round(float(m_t.predict([[tv[-1]]])[0]), 2)
                    gain_t = round((pred_t - current_price) / current_price * 100, 2) if current_price else 0
                    ttm_op = {
                        "model":    "TTM_OP",
                        "label":    f"{profit_label} TTM",
                        "equation": f"Price = {b_t} + {c_t} × {profit_label.split()[0]} TTM",
                        "r2":        r2_t,
                        "pred_price": pred_t,
                        "gain_pct":  gain_t,
                        "data": [
                            {
                                "quarter":      str(common_t[i].date()),
                                "ttm_op":       round(float(tv[i]), 2),
                                "price":        round(float(pqv[i]), 2),
                                "price_to_ttm": round(float(pqv[i] / tv[i]), 4) if tv[i] else None,
                                "fair_value":   round(float(m_t.predict([[tv[i]]])[0]), 2),
                                "misprice_pct": round(
                                    (float(pqv[i]) - float(m_t.predict([[tv[i]]])[0]))
                                    / float(m_t.predict([[tv[i]]])[0]) * 100, 1
                                ) if float(m_t.predict([[tv[i]]])[0]) else 0,
                            }
                            for i in range(len(common_t))
                        ],
                        "historical": _historical_misprice(
                            m_t, tv, pqv,
                            [str(d.date()) for d in common_t], "ttm_op"
                        ),
                    }

        # TTM Sales
        if s_row_q:
            sales_q = _extract_quarterly_series(s_row_q, qr_df).sort_index()
            if len(sales_q) >= 4:
                ttm_sv  = [sales_q.iloc[i - 3:i + 1].sum() for i in range(3, len(sales_q))]
                ttm_si  = sales_q.index[3:]
                ttm_ss  = pd.Series(ttm_sv, index=ttm_si)
                comm_ts = sorted(ttm_ss.index.intersection(price_qe.index))
                if len(comm_ts) >= 2:
                    has_ttm_sales = True
                    sv2  = np.array(ttm_ss.loc[comm_ts]).flatten()
                    pv2  = np.array(price_qe.loc[comm_ts]).flatten()
                    m_ts, r2_ts, b_ts, c_ts = _run_lr(sv2, pv2)
                    pred_ts = round(float(m_ts.predict([[sv2[-1]]])[0]), 2)
                    gain_ts = round((pred_ts - current_price) / current_price * 100, 2) if current_price else 0
                    ttm_sales = {
                        "model":    "TTM_Sales",
                        "label":    f"{s_lbl_q} TTM",
                        "equation": f"Price = {b_ts} + {c_ts} × {s_lbl_q.split()[0]} TTM",
                        "r2":        r2_ts,
                        "pred_price": pred_ts,
                        "gain_pct":  gain_ts,
                        "data": [
                            {
                                "quarter":      str(comm_ts[i].date()),
                                "ttm_sales":    round(float(sv2[i]), 2),
                                "price":        round(float(pv2[i]), 2),
                                "price_to_ttm": round(float(pv2[i] / sv2[i]), 4) if sv2[i] else None,
                                "fair_value":   round(float(m_ts.predict([[sv2[i]]])[0]), 2),
                                "misprice_pct": round(
                                    (float(pv2[i]) - float(m_ts.predict([[sv2[i]]])[0]))
                                    / float(m_ts.predict([[sv2[i]]])[0]) * 100, 1
                                ) if float(m_ts.predict([[sv2[i]]])[0]) else 0,
                            }
                            for i in range(len(comm_ts))
                        ],
                        "historical": _historical_misprice(
                            m_ts, sv2, pv2,
                            [str(d.date()) for d in comm_ts], "ttm_sales"
                        ),
                    }

    # ── Composite ─────────────────────────────────────────────
    # Use the TTM OP block for composite if available, else annual OP
    ttm_for_composite = ttm_op if has_ttm_op else {}
    composite = _compute_composite(yearly_op, yearly_sales if has_sales else {}, ttm_for_composite)

    return {
        # ── Identity ──────────────────────────────────────────
        "ticker":        ticker,
        "current_price": current_price,
        "is_bank":       is_bank,
        "profit_label":  profit_label,

        # ── Composite (top-level) ─────────────────────────────
        "composite_fair_price": composite["composite_fair_price"],
        "composite_gain_pct":   composite["composite_gain_pct"],
        "valuation_bucket":     composite["valuation_bucket"],
        "model_count":          composite["model_count"],

        # ── Annual (year-wise) ────────────────────────────────
        "yearly": {
            "op":    yearly_op,
            "sales": yearly_sales if has_sales else None,
            "has_sales": has_sales,
        },

        # ── Quarterly / TTM ───────────────────────────────────
        "quarterly": {
            "ttm_op":       ttm_op    if has_ttm_op    else None,
            "ttm_sales":    ttm_sales if has_ttm_sales else None,
            "has_ttm_op":   has_ttm_op,
            "has_ttm_sales": has_ttm_sales,
        },
    }


# ─────────────────────────────────────────────────────────────
#  PYDANTIC SCHEMAS
# ─────────────────────────────────────────────────────────────
class FVRequest(BaseModel):
    tickers:              List[str]            = Field(..., min_length=1, max_length=50)
    fy_start:             int                  = Field(2014, ge=2000, le=2030)
    force_scrape:         bool                 = False
    include_other_income: bool                 = True


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@router.get(
    "/fair-value/{ticker}",
    summary="Full fair value — yearly + quarterly + composite",
)
async def get_fair_value(
    ticker:               str,
    fy_start:             int  = Query(2014, ge=2000, le=2030,  description="FY start year"),
    force_scrape:         bool = Query(False,                   description="Bypass disk cache"),
    include_other_income: bool = Query(True,                    description="Add other income (banks)"),
):
    """
    Returns ALL fair-value models for a single ticker:
      • yearly.op         — Annual Operating Profit → Price regression
      • yearly.sales      — Annual Sales → Price regression
      • quarterly.ttm_op  — TTM Operating Profit → Price regression
      • quarterly.ttm_sales — TTM Sales → Price regression
      • composite_*       — R²-weighted blend
    """
    try:
        return await asyncio.to_thread(
            _compute_fair_value, ticker, fy_start, force_scrape, include_other_income
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@router.get(
    "/fair-value/{ticker}/yearly",
    summary="Year-wise fair value only (OP + Sales)",
)
async def get_yearly_fair_value(
    ticker:               str,
    fy_start:             int  = Query(2014, ge=2000, le=2030),
    force_scrape:         bool = Query(False),
    include_other_income: bool = Query(True),
):
    """
    Returns only the year-by-year data:
      yearly.op.data[].year, .op, .fair_value, .misprice_pct, .price
      yearly.sales.data[].year, .sales, .fair_value, .misprice_pct, .price
    """
    try:
        result = await asyncio.to_thread(
            _compute_fair_value, ticker, fy_start, force_scrape, include_other_income
        )
        return {
            "ticker":        result["ticker"],
            "current_price": result["current_price"],
            "yearly":        result["yearly"],
            "composite_fair_price": result["composite_fair_price"],
            "composite_gain_pct":   result["composite_gain_pct"],
            "valuation_bucket":     result["valuation_bucket"],
        }
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get(
    "/fair-value/{ticker}/quarterly",
    summary="Quarter-wise TTM fair value only",
)
async def get_quarterly_fair_value(
    ticker:               str,
    fy_start:             int  = Query(2014, ge=2000, le=2030),
    force_scrape:         bool = Query(False),
    include_other_income: bool = Query(True),
):
    """
    Returns only the TTM quarter-by-quarter data:
      quarterly.ttm_op.data[].quarter, .ttm_op, .fair_value, .price
      quarterly.ttm_sales.data[].quarter, .ttm_sales, .fair_value, .price
    """
    try:
        result = await asyncio.to_thread(
            _compute_fair_value, ticker, fy_start, force_scrape, include_other_income
        )
        return {
            "ticker":        result["ticker"],
            "current_price": result["current_price"],
            "quarterly":     result["quarterly"],
        }
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get(
    "/fair-value/{ticker}/summary",
    summary="Composite + pred prices only (lightweight)",
)
async def get_fair_value_summary(
    ticker:               str,
    fy_start:             int  = Query(2014),
    force_scrape:         bool = Query(False),
    include_other_income: bool = Query(True),
):
    """
    Minimal payload — just the headline numbers.
    Great for dashboards, screeners, or embedding in option tables.
    """
    try:
        r = await asyncio.to_thread(
            _compute_fair_value, ticker, fy_start, force_scrape, include_other_income
        )
        return {
            "ticker":               r["ticker"],
            "current_price":        r["current_price"],
            "composite_fair_price": r["composite_fair_price"],
            "composite_gain_pct":   r["composite_gain_pct"],
            "valuation_bucket":     r["valuation_bucket"],
            "model_count":          r["model_count"],
            "op_fair_price":        r["yearly"]["op"]["pred_price"],
            "op_gain_pct":          r["yearly"]["op"]["gain_pct"],
            "op_r2":                r["yearly"]["op"]["r2"],
            "sales_fair_price":     r["yearly"]["sales"]["pred_price"] if r["yearly"]["sales"] else None,
            "sales_gain_pct":       r["yearly"]["sales"]["gain_pct"]   if r["yearly"]["sales"] else None,
            "sales_r2":             r["yearly"]["sales"]["r2"]         if r["yearly"]["sales"] else None,
            "ttm_op_fair_price":    r["quarterly"]["ttm_op"]["pred_price"]   if r["quarterly"]["ttm_op"]    else None,
            "ttm_op_gain_pct":      r["quarterly"]["ttm_op"]["gain_pct"]     if r["quarterly"]["ttm_op"]    else None,
            "ttm_sales_fair_price": r["quarterly"]["ttm_sales"]["pred_price"] if r["quarterly"]["ttm_sales"] else None,
            "ttm_sales_gain_pct":   r["quarterly"]["ttm_sales"]["gain_pct"]   if r["quarterly"]["ttm_sales"] else None,
        }
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@router.post(
    "/fair-value/batch",
    summary="Batch fair value for multiple tickers",
)
async def batch_fair_value(req: FVRequest):
    """
    Up to 50 tickers in one call.
    Returns: { results: [...], failed: [...], count: int }
    """
    tickers = [t.strip().upper().lstrip("$") for t in req.tickers if t.strip()][:50]
    if not tickers:
        raise HTTPException(400, "No tickers provided.")

    results, failed = [], []
    for tkr in tickers:
        try:
            res = await asyncio.to_thread(
                _compute_fair_value, tkr, req.fy_start, req.force_scrape, req.include_other_income
            )
            results.append(res)
        except Exception as exc:
            failed.append({"ticker": tkr, "error": str(exc)})

    return {"results": results, "failed": failed, "count": len(results)}


@router.delete(
    "/cache/clear",
    summary="Clear cached P&L and quarterly data",
)
async def clear_fv_cache(ticker: Optional[str] = Query(None, description="Clear only this ticker")):
    """
    Clears cached Screener.in CSVs.
    ?ticker=RELIANCE  → clears RELIANCE_pl.csv + RELIANCE_qr.csv only
    (no query param)  → clears ALL _pl.csv and _qr.csv files
    """
    removed = 0
    ticker_upper = ticker.strip().upper() if ticker else None
    for f in os.listdir(DATA_DIR):
        if f.endswith("_pl.csv") or f.endswith("_qr.csv"):
            if ticker_upper and not f.startswith(f"{ticker_upper}_"):
                continue
            try:
                os.remove(os.path.join(DATA_DIR, f))
                removed += 1
            except Exception:
                pass
    return {"removed": removed, "ticker": ticker_upper or "ALL"}
