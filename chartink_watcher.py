"""
Chartink Live Screener Watcher — FastAPI Router
=================================================
Polls the Chartink scan (via chartink_router.run_scan) every 5 minutes,
feeds the returned NSE symbols through the SAME screening pipeline used by
the main 9-EMA Screener tab (ema9_router._run_screen_pipeline — full
EMA9 + trend + Fair-Value enrichment), and persists the latest results to
disk so the frontend "Chartink" tab can poll a cheap GET endpoint instead
of re-running the whole pipeline on every page load.

Daily accumulator: All prime targets seen today are merged into a single
deduplicated list (keyed by ticker). GSheet receives the FULL accumulated
set on every scan via a clean {headers, rows} payload so the sheet is
always an exact snapshot — no duplicate rows across scans.

Mount in main.py:
    from chartink_watcher import router as chartink_watch_router, start_chartink_scheduler
    app.include_router(chartink_watch_router)
    # in startup_event():
    start_chartink_scheduler()

Routes exposed (prefix /api/chartink-screener):
    GET    /api/chartink-screener/results          -> latest saved screener output + metadata
    GET    /api/chartink-screener/today-primes     -> accumulated today's prime targets
    GET    /api/chartink-screener/active-clauses   -> currently selected clause name(s) + all available saved clauses
    POST   /api/chartink-screener/active-clauses   -> persist which saved clause name(s) drive scans
    POST   /api/chartink-screener/scan-now         -> force an immediate scan (?names=a,b or ?clause=<raw>)
    POST   /api/chartink-screener/replay-today     -> re-screen ALL tickers seen in today's log
    DELETE /api/chartink-screener/clear            -> wipe saved results + log file
    GET    /api/chartink-screener/log              -> raw history of which tickers were pulled & when
"""

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import httpx
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from chartink_router import run_scan, DEFAULT_SCAN_CLAUSE, _load_clauses as _load_saved_clauses
from ema9_router import _run_screen_pipeline
from telegram_alert import send_telegram_message, format_prime_targets_message

router = APIRouter(prefix="/api/chartink-screener", tags=["chartink-screener"])

logger = logging.getLogger("chartink_watcher")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR            = "data"
RESULTS_FILE        = os.path.join(DATA_DIR, "chartink_signals.json")
LOG_FILE            = os.path.join(DATA_DIR, "chartink_log.json")
TODAY_PRIMES_FILE   = os.path.join(DATA_DIR, "chartink_today_primes.json")
ACTIVE_CLAUSES_FILE = os.path.join(DATA_DIR, "chartink_active_clauses.json")
SCAN_INTERVAL_MIN   = 5
MAX_LOG_ENTRIES     = 500

# Clause name(s) used when nothing has been explicitly selected yet.
# Matches the "ema9_daily_signal" entry chartink_router.py seeds by default,
# so a fresh install behaves exactly like the old hardcoded-clause version.
DEFAULT_ACTIVE_CLAUSE_NAMES = ["ema9_daily_signal"]

GSHEET_WEBHOOK_URL  = "https://script.google.com/macros/s/AKfycbz3Vj2-_xFkRhqXoySwxDyNZralsZ-XuZamHyuffo7INjtuNPNUSt4lxJg0aqOz7EAe/exec"

# GSheet column order — must match Apps Script header
GSHEET_HEADERS = [
    "Scan Time", "Ticker", "Price", "Fair Value",
    "FV Gap %", "Valuation", "Trend", "Candles Ago", "Type", "Interval",
]
GSHEET_KEYS = [
    "scan_time", "ticker", "price", "fair_value",
    "fv_gap_pct", "valuation", "trend", "candles_ago", "type", "interval",
]

os.makedirs(DATA_DIR, exist_ok=True)

_file_lock  = threading.Lock()
_scan_lock  = threading.Lock()   # prevents overlapping scans
_scheduler: Optional[BackgroundScheduler] = None


# ─────────────────────────────────────────────────────────────
#  PERSISTENCE HELPERS
# ─────────────────────────────────────────────────────────────
def _read_json(path: str, default: Any) -> Any:
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: str, data: Any) -> None:
    with _file_lock:
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)


def load_results() -> Dict:
    return _read_json(RESULTS_FILE, {
        "last_run_at":    None,
        "clause_names":   DEFAULT_ACTIVE_CLAUSE_NAMES,
        "symbols_pulled": [],
        "result":         None,
    })


def save_results(payload: Dict) -> None:
    _write_json(RESULTS_FILE, payload)


def load_log() -> List[Dict]:
    return _read_json(LOG_FILE, [])


def append_log(entry: Dict) -> None:
    log = load_log()
    log.append(entry)
    if len(log) > MAX_LOG_ENTRIES:
        log = log[-MAX_LOG_ENTRIES:]
    _write_json(LOG_FILE, log)


# ─────────────────────────────────────────────────────────────
#  ACTIVE CLAUSE SELECTION
#  Which saved clause name(s) (from chartink_router's registry) currently
#  drive both manual "Scan Now" presses and the background scheduler.
#  Persisted to disk so a server restart keeps the user's last choice.
# ─────────────────────────────────────────────────────────────
def _load_active_clause_names() -> List[str]:
    data = _read_json(ACTIVE_CLAUSES_FILE, {"names": DEFAULT_ACTIVE_CLAUSE_NAMES})
    names = data.get("names") or DEFAULT_ACTIVE_CLAUSE_NAMES
    return names


def _save_active_clause_names(names: List[str]) -> None:
    _write_json(ACTIVE_CLAUSES_FILE, {"names": names})


def _resolve_clause_names(names: List[str]) -> Dict[str, str]:
    """
    Map requested clause names to their raw Chartink scan-clause text using
    chartink_router's persistent registry. Unknown names are skipped (logged).
    If nothing resolves (e.g. registry was cleared), fall back to the
    hardcoded DEFAULT_SCAN_CLAUSE so the watcher never goes silent.
    """
    saved = _load_saved_clauses()
    resolved = {n: saved[n] for n in names if n in saved}
    unknown = [n for n in names if n not in saved]
    if unknown:
        logger.warning(f"[Chartink] Unknown clause name(s) skipped: {unknown}")
    if not resolved:
        logger.warning("[Chartink] No valid clause names resolved — falling back to DEFAULT_SCAN_CLAUSE.")
        resolved = {"_default": DEFAULT_SCAN_CLAUSE}
    return resolved


# ─────────────────────────────────────────────────────────────
#  DAILY PRIME ACCUMULATOR
#  Persists {ticker -> signal_dict} for today's date only.
#  Resets automatically when the date changes.
# ─────────────────────────────────────────────────────────────
def _load_today_primes() -> Dict[str, Dict]:
    """Load today's accumulated prime targets. Returns {} if stale (yesterday's data)."""
    stored = _read_json(TODAY_PRIMES_FILE, {})
    today  = datetime.now().strftime("%Y-%m-%d")
    if stored.get("date") != today:
        return {}
    return stored.get("primes", {})   # { ticker: signal_dict }


def _save_today_primes(primes: Dict[str, Dict]) -> None:
    _write_json(TODAY_PRIMES_FILE, {
        "date":   datetime.now().strftime("%Y-%m-%d"),
        "primes": primes,
    })


def _merge_primes(existing: Dict[str, Dict], new_signals: List[Dict]) -> Dict[str, Dict]:
    """
    Merge new prime signals into existing accumulator.
    Later scans WIN on conflict (fresher price/FV data).
    """
    merged = dict(existing)
    for sig in new_signals:
        ticker = sig.get("ticker")
        if ticker:
            merged[ticker] = sig
    return merged


def _get_today_tickers_from_log() -> Set[str]:
    """Return all unique tickers seen in today's chartink_log.json entries."""
    today = datetime.now().strftime("%Y-%m-%d")
    tickers: Set[str] = set()
    for entry in load_log():
        if entry.get("time", "").startswith(today):
            tickers.update(entry.get("symbols", []))
    return tickers


# ─────────────────────────────────────────────────────────────
#  GSHEET LOGGER  (sends accumulated today's primes with header)
# ─────────────────────────────────────────────────────────────
async def _log_primes_to_gsheet(today_primes: Dict[str, Dict], prime_set: Set[str]) -> None:
    """
    Posts ALL of today's accumulated prime targets to GSheet.
    Sends a clean payload:
      {
        "headers": [...],   <- column names; Apps Script writes these as row 1
        "rows":    [{...}]  <- one dict per prime, most undervalued first
      }
    Apps Script must clearContents() then write headers + rows so the
    sheet is always a clean snapshot with no cross-scan duplicates.
    """
    if not today_primes:
        logger.info("[GSheet] No accumulated primes to push — skipping.")
        return

    scan_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Sort by gap_to_fair_pct descending (most undervalued first)
    # gap_to_fair_pct is the correct key returned by _enrich_fair_value()
    sorted_primes = sorted(
        today_primes.values(),
        key=lambda s: float(s.get("gap_to_fair_pct") or s.get("fv_gap_pct") or 0),
        reverse=True,
    )

    rows = []
    for s in sorted_primes:
        fv_gap = s.get("gap_to_fair_pct") or s.get("fv_gap_pct") or ""
        rows.append({
            "scan_time":   scan_time,
            "ticker":      s.get("ticker", ""),
            "price":       s.get("current_price", ""),
            "fair_value":  s.get("composite_fair_price", ""),
            "fv_gap_pct":  fv_gap,
            "valuation":   s.get("valuation_bucket", ""),   # UNDERVALUED / FAIR / OVERVALUED
            "trend":       s.get("trend_regime", ""),
            "candles_ago": s.get("candles_ago", ""),
            "type":        "PRIME" if s.get("ticker") in prime_set else "OTHER",
            "interval":    s.get("interval", ""),
        })

    payload = {
        "headers": GSHEET_HEADERS,   # Apps Script writes these as row 1
        "rows":    rows,              # Apps Script writes these from row 2 onwards
    }

    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            resp = await client.post(
                GSHEET_WEBHOOK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            body = resp.text[:500]
            if resp.status_code == 200:
                # Parse Apps Script JSON response to catch silent errors
                try:
                    import json as _json
                    gs_resp = _json.loads(body)
                    if gs_resp.get("status") == "ok":
                        logger.info(
                            f"[GSheet] Pushed {len(rows)} primes OK "
                            f"(rows_written={gs_resp.get('rows_written')})."
                        )
                    else:
                        logger.error(
                            f"[GSheet] Apps Script returned error: "
                            f"{gs_resp.get('message', body)}"
                        )
                except Exception:
                    # Non-JSON response (old script still deployed) — log raw
                    logger.warning(f"[GSheet] 200 but non-JSON response: {body}")
            else:
                logger.warning(
                    f"[GSheet] HTTP {resp.status_code} "
                    f"(url={resp.url}): {body}"
                )
    except httpx.TimeoutException:
        logger.warning("[GSheet] Timed out — non-blocking, skipping.")
    except Exception as e:
        logger.warning(f"[GSheet] Logging failed (non-blocking): {e}")


# ─────────────────────────────────────────────────────────────
#  CORE SCAN+SCREEN ROUTINE
# ─────────────────────────────────────────────────────────────
async def run_chartink_screen(clauses: Optional[Dict[str, str]] = None, force_fresh: bool = True) -> Dict:
    """
    1. Hit Chartink once per clause to get each clause's current symbol list,
       then union + dedupe across all of them into a single ticker set.
    2. Run that combined symbol set through the full EMA9 + trend + FV pipeline.
    3. Merge any new prime targets into today's accumulator.
    4. Push the full accumulated prime list to GSheet (with header row).
    5. Persist the latest scan result to disk.

    `clauses`: optional {name: scan_clause_text} mapping to run this time.
    If omitted, resolves from the persisted active clause-name selection
    (chartink_active_clauses.json) — this is what the background scheduler
    uses on every 5-min tick.
    """
    if not _scan_lock.acquire(blocking=False):
        raise HTTPException(409, "A Chartink scan is already running — try again shortly.")

    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        resolved = clauses if clauses else _resolve_clause_names(_load_active_clause_names())

        symbol_set: Set[str] = set()
        per_clause_counts: Dict[str, int] = {}
        for name, clause_text in resolved.items():
            try:
                chartink_data = run_scan(clause_text, use_cache=not force_fresh)
                syms = [r["symbol"] for r in chartink_data["results"] if r.get("symbol")]
            except HTTPException as exc:
                logger.error(f"[Chartink] Clause '{name}' failed: {exc.detail}")
                syms = []
            per_clause_counts[name] = len(syms)
            symbol_set.update(syms)

        symbols = sorted(symbol_set)
        clause_names = list(resolved.keys())

        append_log({
            "time":         ts,
            "clause_names": clause_names,
            "per_clause":   per_clause_counts,
            "symbols":      symbols,
            "count":        len(symbols),
        })

        if not symbols:
            payload = {
                "last_run_at":    ts,
                "clause_names":   clause_names,
                "symbols_pulled": [],
                "result": {
                    "signals": [], "prime_targets": [], "other_signals": [],
                    "filtered_by_trend": [], "failed": [], "fv_failures": [],
                    "count": 0, "prime_count": 0, "other_count": 0,
                },
            }
            save_results(payload)
            logger.info("[Chartink] No symbols returned by any clause; nothing to screen.")
            return payload

        logger.info(
            f"[Chartink] Pulled {len(symbols)} unique symbols from "
            f"{len(resolved)} clause(s) {per_clause_counts}: {symbols}"
        )

        result = await _run_screen_pipeline(
            tickers=symbols,
            interval="1d",
            lookback_days=180,
            max_candles_ago=10,
            require_uptrend=True,
        )

        payload = {
            "last_run_at":    ts,
            "clause_names":   clause_names,
            "symbols_pulled": symbols,
            "result":         result,
        }
        save_results(payload)

        # ── Accumulate today's prime targets ──────────────────────────
        prime_targets   = result.get("prime_targets") or []
        prime_set       = {s.get("ticker") for s in prime_targets}

        today_primes    = _load_today_primes()
        today_primes    = _merge_primes(today_primes, prime_targets)
        _save_today_primes(today_primes)

        logger.info(
            f"[Chartink] This scan: {len(prime_targets)} primes | "
            f"Today total: {len(today_primes)} unique primes"
        )

        # ── GSheet: push full accumulated list (with header) ──────────
        await _log_primes_to_gsheet(today_primes, prime_set=set(today_primes.keys()))

        # ── Telegram alert ────────────────────────────────────────────
        if prime_targets:
            msg = format_prime_targets_message(prime_targets, source="Chartink")
            sent = send_telegram_message(msg)
            logger.info(
                f"[Telegram] Prime-target alert {'sent' if sent else 'FAILED'} "
                f"({len(prime_targets)} tickers)."
            )

        return payload

    finally:
        _scan_lock.release()


# ─────────────────────────────────────────────────────────────
#  REPLAY: re-screen ALL unique tickers from today's log
#  Useful on startup to backfill primes from scans before the
#  server started (or after a restart mid-day).
# ─────────────────────────────────────────────────────────────
async def replay_today_log() -> Dict:
    """
    Collects every unique ticker seen in today's chartink_log.json,
    runs the full screening pipeline on them, merges primes into
    the daily accumulator, and pushes to GSheet.
    """
    today_tickers = _get_today_tickers_from_log()
    if not today_tickers:
        return {"status": "no_log_entries_today", "tickers_screened": 0}

    # Filter out index symbols yfinance can't handle
    _SKIP = {"NIFTY", "BANKNIFTY", "CNXMIDCAP", "FINNIFTY"}
    tickers = sorted(today_tickers - _SKIP)

    logger.info(f"[Chartink-Replay] Replaying {len(tickers)} unique tickers from today's log.")

    result = await _run_screen_pipeline(
        tickers=tickers,
        interval="1d",
        lookback_days=180,
        max_candles_ago=10,
        require_uptrend=True,
    )

    prime_targets = result.get("prime_targets") or []

    today_primes  = _load_today_primes()
    today_primes  = _merge_primes(today_primes, prime_targets)
    _save_today_primes(today_primes)

    await _log_primes_to_gsheet(today_primes, prime_set=set(today_primes.keys()))

    logger.info(
        f"[Chartink-Replay] Done. Found {len(prime_targets)} primes from {len(tickers)} tickers. "
        f"Today total: {len(today_primes)}"
    )

    return {
        "status":            "ok",
        "tickers_screened":  len(tickers),
        "primes_this_run":   len(prime_targets),
        "today_total_primes": len(today_primes),
        "result":            result,
    }


# ─────────────────────────────────────────────────────────────
#  SCHEDULER
# ─────────────────────────────────────────────────────────────
def _scheduled_tick():
    import asyncio
    try:
        asyncio.run(run_chartink_screen(force_fresh=True))
    except Exception as exc:
        logger.error(f"[Chartink] Scheduled scan failed: {exc}")


def start_chartink_scheduler():
    global _scheduler
    if _scheduler is not None:
        return _scheduler
    _scheduler = BackgroundScheduler(daemon=True)
    _scheduler.add_job(
        _scheduled_tick,
        trigger=IntervalTrigger(minutes=SCAN_INTERVAL_MIN),
        id="chartink_scan",
        replace_existing=True,
        max_instances=1,
    )
    _scheduler.start()
    logger.info(f"[Chartink] Scheduler started — scanning every {SCAN_INTERVAL_MIN} min.")
    return _scheduler


def stop_chartink_scheduler():
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None


# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────
@router.get("/results")
async def get_chartink_results():
    """Return the most recently saved Chartink-sourced screener output."""
    return load_results()


@router.get("/today-primes")
async def get_today_primes():
    """Return today's full accumulated prime targets (deduplicated across all scans)."""
    primes = _load_today_primes()
    return {
        "date":   datetime.now().strftime("%Y-%m-%d"),
        "count":  len(primes),
        "primes": list(primes.values()),
    }


@router.get("/active-clauses")
async def get_active_clauses():
    """
    Currently selected clause name(s) driving 'Scan Now' and the background
    scheduler, plus the full set of available saved clauses (from chartink_router's
    registry) so the frontend can render a checkbox list without a second round-trip.
    """
    return {
        "names":     _load_active_clause_names(),
        "available": _load_saved_clauses(),
    }


class ActiveClausesUpdate(BaseModel):
    names: List[str]


@router.post("/active-clauses")
async def set_active_clauses(req: ActiveClausesUpdate):
    """
    Persist which saved clause name(s) should drive 'Scan Now' and the
    background scheduler going forward. Validates names against the
    chartink_router clause registry.
    """
    if not req.names:
        raise HTTPException(400, "names must not be empty")
    saved = _load_saved_clauses()
    unknown = [n for n in req.names if n not in saved]
    if unknown:
        raise HTTPException(404, f"Unknown clause name(s): {unknown}")
    _save_active_clause_names(req.names)
    return {"ok": True, "names": req.names}


@router.post("/scan-now")
async def scan_now(names: Optional[str] = None, clause: Optional[str] = None):
    """
    Force an immediate Chartink pull + full screen, bypassing the 5-min schedule.

    - ?names=ema9_5min_breakout,ema9_daily_signal
        Run these saved clause names (union of their symbol lists), AND persist
        this as the new active selection so the background scheduler picks it
        up on its next tick too.
    - ?clause=<raw scan_clause text>
        One-off ad-hoc raw clause, NOT added to the registry and NOT persisted
        as the active selection — useful for a quick test run.
    - (neither)
        Re-run whatever is currently the persisted active selection.
    """
    if clause:
        return await run_chartink_screen({"_adhoc": clause}, force_fresh=True)

    if names:
        clause_names = [n.strip() for n in names.split(",") if n.strip()]
        if not clause_names:
            raise HTTPException(400, "names must contain at least one clause name")
        resolved = _resolve_clause_names(clause_names)
        _save_active_clause_names(clause_names)
        return await run_chartink_screen(resolved, force_fresh=True)

    return await run_chartink_screen(None, force_fresh=True)


@router.post("/replay-today")
async def replay_today():
    """
    Re-screen ALL unique tickers seen in today's Chartink log.
    Use this on server startup or after a restart to backfill today's primes.
    """
    return await replay_today_log()


@router.delete("/clear")
async def clear_chartink_results():
    """Clear saved results, today's prime accumulator, and the pull log."""
    save_results({
        "last_run_at":    None,
        "clause_names":   _load_active_clause_names(),
        "symbols_pulled": [],
        "result":         None,
    })
    _write_json(TODAY_PRIMES_FILE, {"date": None, "primes": {}})
    _write_json(LOG_FILE, [])
    return {"ok": True}


@router.get("/log")
async def get_chartink_log():
    """Raw history of each Chartink pull (timestamp, clause, symbols returned)."""
    return {"log": load_log()}


@router.post("/test-telegram")
async def test_telegram():
    """Send a test message to verify Telegram config."""
    ok = send_telegram_message("✅ Chartink screener Telegram alert is configured correctly.")
    if not ok:
        raise HTTPException(
            500,
            "Telegram send failed — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars."
        )
    return {"ok": True, "message": "Test message sent."}