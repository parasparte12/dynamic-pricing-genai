#!/usr/bin/env bash
# Run this ON THE SERVER after pushing new code to GitHub, to bring the
# deployed app up to date. Not run automatically by anything -- there is no
# CI/CD in this project; this is the manual "pull latest, rebuild, restart" step.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> Pulling latest code..."
git pull

# fastapi and streamlit are built from the SAME image (see Dockerfile) -- a
# Python code change or a requirements.txt change affects both, so both are
# rebuilt together here. This is intentionally NOT `docker compose up -d --build`
# (which would rebuild/restart everything, including ollama and nginx): those two
# use unmodified official images and don't need touching for an application
# code change, and restarting nginx/ollama unnecessarily would cause a brief
# outage and, for ollama, re-run its healthcheck warmup for no reason.
echo "==> Rebuilding fastapi and streamlit..."
docker compose build fastapi streamlit

echo "==> Restarting fastapi and streamlit..."
docker compose up -d --no-deps fastapi streamlit

echo "==> Done. ollama and nginx were left running untouched."
echo "    (If nginx/nginx.conf itself changed, run: docker compose up -d --no-deps nginx)"
echo "    (If the base Dockerfile/system deps changed, a full 'docker compose build' is safest.)"
