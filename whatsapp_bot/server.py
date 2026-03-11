"""
WhatsApp webhook server supporting both Meta Cloud API and Twilio.

IPC with the Claude Code session via files:
  queue/incoming/<id>.json   <- server writes when WhatsApp message arrives
  queue/outgoing/<id>.txt    <- Claude Code session writes its response
  queue/processed/<id>.json  <- server moves processed incoming files here

Configuration is read from config.json in this directory.

Run:
  cd whatsapp_bot && python server.py
"""

import json
import os
import time
import uuid
import threading
from datetime import datetime
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import PlainTextResponse, JSONResponse
import uvicorn

# -- Config -------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config() -> dict:
    if not CONFIG_PATH.exists():
        raise RuntimeError(
            f"config.json not found at {CONFIG_PATH}. "
            "Run the /whatsapp skill in Claude Code to set up."
        )
    return json.loads(CONFIG_PATH.read_text())


CONFIG = load_config()
PROVIDER = CONFIG.get("provider", "meta")

# Build set of all allowed phone numbers from all user profiles
ALLOWED_NUMBERS: set[str] = set()
for user in CONFIG.get("users", {}).values():
    phone = user.get("phone", "")
    if phone:
        ALLOWED_NUMBERS.add(phone)
        # Also allow with whatsapp: prefix (Twilio format)
        ALLOWED_NUMBERS.add(f"whatsapp:+{phone}")

QUEUE_DIR = Path(CONFIG.get("queue_dir", Path(__file__).parent / "queue"))
RESPONSE_TIMEOUT = int(CONFIG.get("response_timeout", 600))
PROGRESS_INTERVAL = int(CONFIG.get("progress_interval", 30))
PORT = int(CONFIG.get("port", 8000))

INCOMING_DIR = QUEUE_DIR / "incoming"
OUTGOING_DIR = QUEUE_DIR / "outgoing"
PROCESSED_DIR = QUEUE_DIR / "processed"

for d in (INCOMING_DIR, OUTGOING_DIR, PROCESSED_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -- Sending ------------------------------------------------------------------


def send_wa(to: str, body: str) -> None:
    """Send a WhatsApp message using the configured provider."""
    if PROVIDER == "meta":
        _send_meta(to, body)
    elif PROVIDER == "twilio":
        _send_twilio(to, body)
    else:
        print(f"[send_wa] Unknown provider: {PROVIDER}")


def _send_meta(to: str, body: str) -> None:
    meta_cfg = CONFIG.get("meta", {})
    phone_id = meta_cfg.get("phone_number_id", "")
    token = meta_cfg.get("access_token", "")
    url = f"https://graph.facebook.com/v21.0/{phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    # Strip whatsapp: prefix if present
    to_clean = to.replace("whatsapp:+", "").replace("whatsapp:", "").lstrip("+")
    for i in range(0, max(len(body), 1), 4000):
        chunk = body[i : i + 4000] or "[empty]"
        payload = {
            "messaging_product": "whatsapp",
            "to": to_clean,
            "type": "text",
            "text": {"body": chunk},
        }
        resp = httpx.post(url, headers=headers, json=payload, timeout=30)
        if resp.status_code != 200:
            print(f"[send_meta] Error {resp.status_code}: {resp.text}")


def _send_twilio(to: str, body: str) -> None:
    from twilio.rest import Client as TwilioClient

    twilio_cfg = CONFIG.get("twilio", {})
    client = TwilioClient(twilio_cfg["account_sid"], twilio_cfg["auth_token"])
    wa_number = twilio_cfg.get("wa_number", "whatsapp:+14155238886")
    # Ensure whatsapp: prefix
    if not to.startswith("whatsapp:"):
        to = f"whatsapp:+{to}"
    for i in range(0, max(len(body), 1), 1500):
        chunk = body[i : i + 1500] or "[empty]"
        client.messages.create(from_=wa_number, to=to, body=chunk)


# -- Helpers ------------------------------------------------------------------


def _normalize_sender(sender: str) -> str:
    """Normalize phone number to digits-only format."""
    return sender.replace("whatsapp:+", "").replace("whatsapp:", "").lstrip("+")


def _is_allowed(sender: str) -> bool:
    normalized = _normalize_sender(sender)
    return normalized in ALLOWED_NUMBERS or sender in ALLOWED_NUMBERS


def _enqueue_message(sender: str, body: str) -> str:
    """Write an incoming message to the queue. Returns the message ID."""
    msg_id = f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:6]}"
    incoming = {
        "id": msg_id,
        "from": _normalize_sender(sender),
        "body": body,
        "timestamp": datetime.utcnow().isoformat(),
    }
    (INCOMING_DIR / f"{msg_id}.json").write_text(json.dumps(incoming, indent=2))
    return msg_id


# -- App ----------------------------------------------------------------------

app = FastAPI(title="WA-Claude Skill Server")


@app.get("/health")
async def health():
    pending = list(INCOMING_DIR.glob("*.json"))
    return {
        "status": "ok",
        "provider": PROVIDER,
        "active_user": CONFIG.get("active_user", ""),
        "pending_messages": len(pending),
    }


@app.get("/config")
async def get_config():
    """Show current configuration (without secrets)."""
    safe = {
        "provider": PROVIDER,
        "active_user": CONFIG.get("active_user", ""),
        "users": {
            name: {"name": u.get("name", name), "phone": u.get("phone", "")}
            for name, u in CONFIG.get("users", {}).items()
        },
    }
    return safe


# -- Meta webhook endpoints ---------------------------------------------------


@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta webhook verification (GET with hub.challenge)."""
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    meta_cfg = CONFIG.get("meta", {})
    verify_token = meta_cfg.get("verify_token", "")

    if mode == "subscribe" and token == verify_token:
        print("[webhook] Meta verification successful")
        return PlainTextResponse(content=challenge, status_code=200)

    raise HTTPException(status_code=403, detail="Verification failed")


@app.post("/webhook")
async def webhook(request: Request):
    """Handle incoming messages from either Meta or Twilio."""
    content_type = request.headers.get("content-type", "")

    if "application/json" in content_type:
        # Meta Cloud API sends JSON
        return await _handle_meta_webhook(request)
    elif "application/x-www-form-urlencoded" in content_type:
        # Twilio sends form-encoded data
        return await _handle_twilio_webhook(request)
    else:
        # Try JSON first, fall back to form
        try:
            return await _handle_meta_webhook(request)
        except Exception:
            return await _handle_twilio_webhook(request)


async def _handle_meta_webhook(request: Request) -> JSONResponse:
    data = await request.json()

    if data.get("object") != "whatsapp_business_account":
        return JSONResponse(content={"status": "ignored"}, status_code=200)

    for entry in data.get("entry", []):
        for change in entry.get("changes", []):
            value = change.get("value", {})
            for msg in value.get("messages", []):
                if msg.get("type") != "text":
                    continue
                sender = msg["from"]
                if not _is_allowed(sender):
                    print(f"[webhook] Blocked message from {sender}")
                    continue
                body = msg["text"]["body"].strip()
                msg_id = _enqueue_message(sender, body)
                threading.Thread(
                    target=_wait_and_reply, args=(msg_id, sender), daemon=True
                ).start()

    return JSONResponse(content={"status": "ok"}, status_code=200)


async def _handle_twilio_webhook(request: Request) -> PlainTextResponse:
    form = await request.form()
    sender = form.get("From", "")
    body = str(form.get("Body", "")).strip()

    if not _is_allowed(sender):
        raise HTTPException(status_code=403, detail="Number not allowed")

    msg_id = _enqueue_message(sender, body)
    threading.Thread(
        target=_wait_and_reply, args=(msg_id, _normalize_sender(sender)), daemon=True
    ).start()

    # Return empty TwiML (no ACK message, saves quota)
    return PlainTextResponse(
        content='<?xml version="1.0" encoding="UTF-8"?><Response></Response>',
        media_type="application/xml",
    )


# -- Response waiter ----------------------------------------------------------


def _wait_and_reply(msg_id: str, sender: str) -> None:
    """Block until Claude writes the response file, then send it."""
    outgoing_file = OUTGOING_DIR / f"{msg_id}.txt"
    incoming_file = INCOMING_DIR / f"{msg_id}.json"
    elapsed = 0

    while elapsed < RESPONSE_TIMEOUT:
        if outgoing_file.exists():
            reply = outgoing_file.read_text().strip()
            send_wa(sender, reply or "[Claude returned empty response]")
            outgoing_file.unlink(missing_ok=True)
            if incoming_file.exists():
                incoming_file.rename(PROCESSED_DIR / incoming_file.name)
            return

        time.sleep(PROGRESS_INTERVAL)
        elapsed += PROGRESS_INTERVAL

    send_wa(sender, f"[Timed out after {RESPONSE_TIMEOUT}s]")


if __name__ == "__main__":
    print(f"Provider: {PROVIDER} | Users: {list(CONFIG.get('users', {}).keys())}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
