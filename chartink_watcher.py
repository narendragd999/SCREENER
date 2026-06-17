"""
Chartink Live Screener Watcher — FastAPI Router
=================================================
Polls the Chartink scan (via chartink_router.run_scan) every 5 minutes,
feeds the returned NSE symbols through the SAME screening pipeline used by
the main 9-EMA Screener tab (ema9_router._run_screen_pipeline — full
EMA9 + trend + Fair-Value enrichment), and persists the latest results to
disk so the frontend "Chartink" tab can poll a cheap GET endpoint instead
of re-running the whole pipeline on every page load.

Mount in main.py:
    from chartink_watcher import router as chartink_watch_router, start_chartink_scheduler
    app.include_router(chartink_watch_router)
    # in startup_event():
    start_chartink_scheduler()

Routes exposed (prefix /api/chartink-screener):
    GET    /api/chartink-screener/results     -> latest saved screener output + metadata
    POST   /api/chartink-screener/scan-now    -> force an immediate scan (blocking call, returns result)
    DELETE /api/chartink-screener/clear       -> wipe saved results + log file
    GET    /api/chartink-screener/log         -> raw history of which tickers were pulled & when
"""

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import APIRouter, HTTPException

from chartink_router import run_scan, DEFAULT_SCAN_CLAUSE
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
DATA_DIR          = "data"
RESULTS_FILE      = os.path.join(DATA_DIR, "chartink_signals.json")
LOG_FILE          = os.path.join(DATA_DIR, "chartink_log.json")
SCAN_INTERVAL_MIN = 5
MAX_LOG_ENTRIES   = 500

os.makedirs(DATA_DIR, exist_ok=True)

_file_lock = threading.Lock()
_scan_lock = threading.Lock()   # prevents overlapping scans (scheduler tick vs manual scan-now)
_scheduler: Optional[BackgroundScheduler] = None


# ─────────────────────────────────────────────────────────────
#  PERSISTENCE HELPERS (plain JSON files, consistent with rest of app)
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
        "last_run_at":   None,
        "scan_clause":   DEFAULT_SCAN_CLAUSE,
        "symbols_pulled": [],
        "result":        None,
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
#  CORE SCAN+SCREEN ROUTINE
# ─────────────────────────────────────────────────────────────
async def run_chartink_screen(scan_clause: str = DEFAULT_SCAN_CLAUSE, force_fresh: bool = True) -> Dict:
    """
    1. Hit Chartink to get the current symbol list for scan_clause.
    2. Run those symbols through the EXACT SAME pipeline as the CSV-upload
       screener tab (EMA9 breakdown/breakout + trend filter + Fair Value).
    3. Persist the combined result + an append-only log of each pull.
    """
    if not _scan_lock.acquire(blocking=False):
        raise HTTPException(409, "A Chartink scan is already running — try again shortly.")

    try:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        chartink_data = run_scan(scan_clause, use_cache=not force_fresh)
        symbols = [r["symbol"] for r in chartink_data["results"] if r.get("symbol")]

        append_log({
            "time":        ts,
            "scan_clause": scan_clause,
            "symbols":     symbols,
            "count":       len(symbols),
        })

        if not symbols:
            payload = {
                "last_run_at":    ts,
                "scan_clause":    scan_clause,
                "symbols_pulled": [],
                "result":         {
                    "signals": [], "prime_targets": [], "other_signals": [],
                    "filtered_by_trend": [], "failed": [], "fv_failures": [],
                    "count": 0, "prime_count": 0, "other_count": 0,
                },
            }
            save_results(payload)
            logger.info("[Chartink] No symbols returned by scan; nothing to screen.")
            return payload

        logger.info(f"[Chartink] Pulled {len(symbols)} symbols: {symbols}")

        # Same screening pipeline as the main Screener tab (CSV upload route)
        result = await _run_screen_pipeline(
            tickers=symbols,
            interval="1d",
            lookback_days=180,
            max_candles_ago=10,
            require_uptrend=True,
        )

        payload = {
            "last_run_at":    ts,
            "scan_clause":    scan_clause,
            "symbols_pulled": symbols,
            "result":         result,
        }
        save_results(payload)

        # Telegram alert — fires every scan when prime targets exist (repeats included)
        prime_targets = result.get("prime_targets") or []
        if prime_targets:
            msg = format_prime_targets_message(prime_targets, source="Chartink")
            sent = send_telegram_message(msg)
            logger.info(f"[Telegram] Prime-target alert {'sent' if sent else 'FAILED'} "
                        f"({len(prime_targets)} tickers).")

        return payload

    finally:
        _scan_lock.release()


# ─────────────────────────────────────────────────────────────
#  SCHEDULER (every 5 minutes, market-hours agnostic — Chartink scan
#  itself naturally returns nothing outside trading hours)
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


@router.post("/scan-now")
async def scan_now(clause: Optional[str] = None):
    """Force an immediate Chartink pull + full screen, bypassing the 5-min schedule."""
    return await run_chartink_screen(clause or DEFAULT_SCAN_CLAUSE, force_fresh=True)


@router.delete("/clear")
async def clear_chartink_results():
    """Clear saved results and the pull log (does not stop the scheduler)."""
    save_results({
        "last_run_at": None,
        "scan_clause": DEFAULT_SCAN_CLAUSE,
        "symbols_pulled": [],
        "result": None,
    })
    _write_json(LOG_FILE, [])
    return {"ok": True}


@router.get("/log")
async def get_chartink_log():
    """Raw history of each Chartink pull (timestamp, clause, symbols returned)."""
    return {"log": load_log()}


@router.post("/test-telegram")
async def test_telegram():
    """Send a test message to verify TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID are configured correctly."""
    ok = send_telegram_message("✅ Chartink screener Telegram alert is configured correctly.")
    if not ok:
        raise HTTPException(
            500,
            "Telegram send failed — check TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars and server logs."
        )
    return {"ok": True, "message": "Test message sent."}