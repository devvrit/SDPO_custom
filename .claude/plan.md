# WhatsApp Skill Redesign Plan

## Goal
Make the WhatsApp skill shareable, with interactive setup, dual provider support (Twilio / Meta), user profiles, and dynamic path detection.

## Architecture

### Config-driven approach
Replace `.env` with `whatsapp_bot/config.json`:
```json
{
  "provider": "meta",
  "active_user": "devvrit",
  "users": {
    "devvrit": { "name": "Devvrit", "phone": "15109739281" }
  },
  "meta": {
    "phone_number_id": "...",
    "access_token": "...",
    "verify_token": "..."
  },
  "twilio": {
    "account_sid": "...",
    "auth_token": "...",
    "wa_number": "whatsapp:+14155238886"
  },
  "port": 8000,
  "response_timeout": 600
}
```

## Files to change

### 1. `whatsapp_bot/server.py` — Rewrite
- Read `config.json` instead of `.env`
- Support both backends via a `send_wa()` dispatcher
- Handle both Twilio POST (form-encoded) and Meta POST (JSON) webhook formats on the same `/webhook` endpoint
- Handle Meta GET verification on `/webhook`
- Allow messages from ALL configured user phone numbers (not just active user)
- Add `/config` GET endpoint to show current provider + active user

### 2. `~/.claude/skills/whatsapp/SKILL.md` — Rewrite
**First-run setup wizard** (when `config.json` doesn't exist):
1. Auto-detect `whatsapp_bot/` directory relative to working dir
2. Ask user: Twilio or Meta Cloud API?
3. Walk through credentials for chosen provider
4. Ask for user name + phone number
5. Install deps (`pip install -r requirements.txt`)
6. Start server + ngrok
7. Guide webhook URL configuration (Twilio console or Meta developer portal)
8. Save `config.json`

**Returning user** (config exists):
- Check for special commands before entering the poll loop:
  - `/add-user` — add a new user profile
  - `/switch-user <name>` — change active user
  - `/switch-provider` — switch between Twilio/Meta (prompt for credentials if not configured)
  - `/wa-status` — show current config (provider, active user, all users)
- Start server + ngrok if not running
- Enter normal polling loop

**Polling loop** — same as current (check incoming dir, process, write outgoing, repeat)

### 3. `whatsapp_bot/requirements.txt` — Update
```
fastapi>=0.111.0
uvicorn[standard]>=0.29.0
twilio>=9.0.0
httpx>=0.27.0
python-dotenv>=1.0.0
python-multipart>=0.0.9
```

### 4. `whatsapp_bot/.env.example` — Delete (replaced by config.json)

### 5. `whatsapp_bot/config.example.json` — New
Example config for reference.

### 6. `whatsapp_bot/start.sh` — Update
Read from config.json, print current provider/user.

## Key design decisions
- **Single server handles both providers** — no need to restart when switching
- **All user phones are allowed** — the "active_user" only controls who the polling loop responds to (for the SKILL.md side)
- **Paths are relative** — SKILL.md detects `whatsapp_bot/` in the current working directory
- **Config.json is gitignored** — contains secrets; `config.example.json` is committed
