"""
Chartink Screener Client — FastAPI Router
==========================================
Wraps Chartink's public screener endpoint (chartink.com/screener/process)
behind a clean, cached, retry-safe FastAPI router.

Mount in main.py:
    from chartink_router import router as chartink_router
    app.include_router(chartink_router)

Routes exposed:
    GET  /api/chartink/scan            -> run the default (or ?clause=) scan, cached
    POST /api/chartink/scan            -> run a scan with a custom scan_clause in body
    GET  /api/chartink/symbols         -> just the list of NSE symbols from default scan
    DELETE /api/chartink/cache         -> clear cached results
"""

import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from urllib.parse import unquote

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/chartink", tags=["chartink"])

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
CHARTINK_BASE     = "https://chartink.com"
CHARTINK_PROCESS  = f"{CHARTINK_BASE}/screener/process"
DEFAULT_SCAN_CLAUSE = (
    "( {33489} ( [0] 5 minute close > [0] 5 minute ema( close,9 ) "
    "and [ -1 ] 5 minute close <= [ -1 ] 5 minute ema( close,9 ) ) )"
)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

CACHE_TTL_SECS    = 30     # avoid hammering chartink on rapid re-polls
SESSION_TTL_SECS  = 20 * 60  # refresh XSRF/session cookie periodically
REQUEST_TIMEOUT   = 15

# ─────────────────────────────────────────────────────────────
#  SESSION MANAGEMENT (thread-safe, lazily refreshed)
# ─────────────────────────────────────────────────────────────
_session: Optional[requests.Session] = None
_session_created: Optional[datetime] = None
_session_lock = threading.Lock()

_cache: Dict[str, Dict[str, Any]] = {}   # clause -> {"ts": float, "data": [...]}
_cache_lock = threading.Lock()


def _new_session() -> requests.Session:
    s = requests.Session()
    r = s.get(CHARTINK_BASE, headers={"User-Agent": UA}, timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    return s


def _get_session(force: bool = False) -> requests.Session:
    """Return a live session, creating/refreshing it if stale or forced."""
    global _session, _session_created
    with _session_lock:
        stale = (
            _session is None
            or _session_created is None
            or (datetime.now() - _session_created).total_seconds() > SESSION_TTL_SECS
        )
        if force or stale:
            _session = _new_session()
            _session_created = datetime.now()
        return _session


def _xsrf_headers(session: requests.Session) -> Dict[str, str]:
    token = session.cookies.get("XSRF-TOKEN")
    if not token:
        raise HTTPException(502, "Chartink did not return an XSRF-TOKEN cookie")
    token = unquote(token)
    return {
        "content-type":     "application/json",
        "x-requested-with": "XMLHttpRequest",
        "x-xsrf-token":     token,
        "referer":          f"{CHARTINK_BASE}/screener",
        "origin":           CHARTINK_BASE,
        "user-agent":       UA,
    }


def _parse_rows(raw: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Chartink returns {"data": [{"sr": .., "nsecode": "RELIANCE", "close": .., ...}, ...]}"""
    rows = raw.get("data", [])
    out = []
    for row in rows:
        out.append({
            "symbol":      row.get("nsecode") or row.get("bsecode") or row.get("name"),
            "name":        row.get("name"),
            "close":       row.get("close"),
            "per_chg":     row.get("per_chg"),
            "volume":      row.get("volume"),
            "raw":         row,
        })
    return out


def _post_scan(scan_clause: str, retry_on_auth_fail: bool = True) -> List[Dict[str, Any]]:
    session = _get_session()
    headers = _xsrf_headers(session)
    payload = {"scan_clause": scan_clause, "debug_clause": "", "column_clause": ""}

    resp = session.post(CHARTINK_PROCESS, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

    # Session/XSRF likely expired -> refresh once and retry
    if resp.status_code in (401, 403, 419) and retry_on_auth_fail:
        session = _get_session(force=True)
        headers = _xsrf_headers(session)
        resp = session.post(CHARTINK_PROCESS, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)

    if resp.status_code != 200:
        raise HTTPException(
            502,
            f"Chartink returned status {resp.status_code}: {resp.text[:300]}"
        )

    try:
        raw = resp.json()
    except ValueError:
        raise HTTPException(502, f"Chartink returned non-JSON response: {resp.text[:300]}")

    if "data" not in raw:
        # Chartink returns {"errors": "..."} on a malformed scan_clause
        raise HTTPException(400, f"Chartink error: {raw}")

    return _parse_rows(raw)


def run_scan(scan_clause: str = DEFAULT_SCAN_CLAUSE, use_cache: bool = True) -> Dict[str, Any]:
    """Public function other modules (e.g. main.py) can call directly, with caching."""
    cache_key = scan_clause

    if use_cache:
        with _cache_lock:
            cached = _cache.get(cache_key)
            if cached and (time.time() - cached["ts"]) < CACHE_TTL_SECS:
                return {"results": cached["data"], "count": len(cached["data"]), "cached": True}

    results = _post_scan(scan_clause)

    with _cache_lock:
        _cache[cache_key] = {"ts": time.time(), "data": results}

    return {"results": results, "count": len(results), "cached": False}


# ─────────────────────────────────────────────────────────────
#  SCHEMAS
# ─────────────────────────────────────────────────────────────
class ScanRequest(BaseModel):
    scan_clause: str
    use_cache:   bool = True


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@router.get("/scan")
async def get_scan(clause: Optional[str] = None, use_cache: bool = True):
    """
    Run the Chartink screener scan.
    Pass ?clause=<scan_clause> to override the default 5-min EMA9-breakout clause.
    Set ?use_cache=false to force a fresh hit against chartink.com.
    """
    scan_clause = clause or DEFAULT_SCAN_CLAUSE
    return run_scan(scan_clause, use_cache=use_cache)


@router.post("/scan")
async def post_scan(req: ScanRequest):
    """Run the Chartink screener scan with a custom scan_clause in the request body."""
    if not req.scan_clause.strip():
        raise HTTPException(400, "scan_clause must not be empty")
    return run_scan(req.scan_clause, use_cache=req.use_cache)


@router.get("/symbols")
async def get_symbols(clause: Optional[str] = None):
    """Convenience route: just the NSE symbol list from the scan."""
    data = run_scan(clause or DEFAULT_SCAN_CLAUSE)
    symbols = [r["symbol"] for r in data["results"] if r.get("symbol")]
    return {"symbols": symbols, "count": len(symbols), "cached": data["cached"]}


@router.delete("/cache")
async def clear_cache():
    """Clear cached scan results, forcing the next call to hit chartink.com fresh."""
    with _cache_lock:
        _cache.clear()
    return {"ok": True}
