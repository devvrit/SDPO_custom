#!/usr/bin/env bash
# Start the WhatsApp webhook server.
set -e
cd "$(dirname "$0")"

if [ ! -f config.json ]; then
    echo "ERROR: config.json not found."
    echo "Run the /whatsapp skill in Claude Code for guided setup,"
    echo "or copy config.example.json to config.json and fill in your values."
    exit 1
fi

PROVIDER=$(python3 -c "import json; print(json.load(open('config.json'))['provider'])")
ACTIVE=$(python3 -c "import json; print(json.load(open('config.json')).get('active_user',''))")
PORT=$(python3 -c "import json; print(json.load(open('config.json')).get('port', 8000))")

echo "=== WhatsApp Relay Server ==="
echo "Provider    : $PROVIDER"
echo "Active user : $ACTIVE"
echo "Port        : $PORT"
echo ""
echo "Expose with:  ngrok http $PORT"
echo "Then set webhook to: https://<ngrok-id>.ngrok-free.app/webhook"
echo ""

python server.py
