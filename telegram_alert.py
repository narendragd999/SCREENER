"""
Telegram Alert Helper
======================
Thin wrapper around the Telegram Bot API's sendMessage endpoint, reusing
the SAME config.json (telegram_bot_token / telegram_chat_id) that main.py's
own send_telegram() already reads — so there's a single source of truth for
credentials and you only ever configure them in one place.

CONFIG SOURCE:
    config.json -> { "telegram_bot_token": "...", "telegram_chat_id": "..." }
    (Same keys main.py's DEFAULT_CONFIG already defines and load_config() loads.)

If either value is empty/missing, send_telegram_message() logs a warning
and returns False instead of raising — a missing/misconfigured Telegram
setup never breaks the screener itself.
"""

import json
import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger("telegram_alert")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

TELEGRAM_API_BASE = "https://api.telegram.org"
REQUEST_TIMEOUT    = 10
CONFIG_FILE        = "config.json"   # same file main.py reads/writes via load_config()/save_config()


def _load_telegram_creds() -> Optional[tuple]:
    """
    Read telegram_bot_token / telegram_chat_id straight from config.json.
    Re-reads on every call (cheap, small file) so a config update made via
    the main app's /api/config route takes effect immediately without restart.
    """
    if not os.path.exists(CONFIG_FILE):
        logger.warning(f"[Telegram] {CONFIG_FILE} not found — skipping alert.")
        return None
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as exc:
        logger.error(f"[Telegram] Failed to read {CONFIG_FILE}: {exc}")
        return None

    token   = str(cfg.get("telegram_bot_token") or "").strip()
    chat_id = str(cfg.get("telegram_chat_id") or "").strip()
    if not token or not chat_id:
        logger.warning(
            "[Telegram] telegram_bot_token / telegram_chat_id empty in config.json — skipping alert."
        )
        return None
    return token, chat_id


def send_telegram_message(text: str, parse_mode: str = "Markdown") -> bool:
    """
    Send a message to the chat configured in config.json.
    Chunks at 4096 chars (Telegram's hard limit), matching main.py's send_telegram().
    Returns True if all chunks sent successfully, False otherwise (never raises).
    """
    creds = _load_telegram_creds()
    if not creds:
        return False
    token, chat_id = creds

    url = f"{TELEGRAM_API_BASE}/bot{token}/sendMessage"
    ok_all = True
    for i in range(0, len(text), 4096):
        chunk = text[i:i + 4096].strip()
        if not chunk:
            continue
        try:
            resp = requests.post(
                url,
                json={"chat_id": chat_id, "text": chunk, "parse_mode": parse_mode},
                timeout=REQUEST_TIMEOUT,
            )
            if resp.status_code != 200:
                logger.error(f"[Telegram] sendMessage failed ({resp.status_code}): {resp.text[:300]}")
                ok_all = False
        except requests.RequestException as exc:
            logger.error(f"[Telegram] sendMessage request error: {exc}")
            ok_all = False
        time.sleep(0.3)

    return ok_all


def format_prime_targets_message(prime_targets: list, source: str = "Chartink") -> str:
    """
    Build a compact Markdown-formatted Telegram message for a list of
    PRIME TARGET signals (undervalued + uptrend), matching the screener's
    own field names (ticker, current_price, target_3pct, composite_fair_price,
    gap_to_fair_pct, trend_regime). Uses Markdown (*bold*) to match main.py's
    existing send_telegram() parse_mode.
    """
    if not prime_targets:
        return ""

    lines = [f"🎯 *PRIME TARGETS* — {source} ({len(prime_targets)})\n"]
    for s in prime_targets:
        ticker = s.get("ticker") or "?"
        price  = s.get("current_price")
        target = s.get("target_3pct")
        fv     = s.get("composite_fair_price")
        gap    = s.get("gap_to_fair_pct")
        trend  = s.get("trend_regime") or "—"

        price_s  = f"Rs{price:,.2f}"  if price  is not None else "—"
        target_s = f"Rs{target:,.2f}" if target is not None else "—"
        fv_s     = f"Rs{fv:,.2f}"     if fv     is not None else "—"
        gap_s    = f"{'+' if (gap or 0) >= 0 else ''}{gap:.1f}%" if gap is not None else "—"

        lines.append(
            f"• *{ticker}*  {price_s} → 🎯{target_s}\n"
            f"   FV: {fv_s}  (gap {gap_s})  📈 {trend}"
        )

    return "\n".join(lines)