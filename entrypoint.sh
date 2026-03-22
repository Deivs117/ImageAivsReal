#!/usr/bin/env bash
set -euo pipefail

# Configurable timeout (segundos). Puedes sobrescribir con MODEL_STARTUP_TIMEOUT env var.
TIMEOUT="${MODEL_STARTUP_TIMEOUT:-1800}"  # default 1800s = 30 minutos

echo "Starting inference server..."
/opt/venv/bin/python -u service/inference_server.py &
SERVER_PID=$!

# Señales
term_handler() {
  echo "Shutting down..."
  kill -TERM "$SERVER_PID" 2>/dev/null || true
  wait "$SERVER_PID" 2>/dev/null || true
  exit 0
}
trap 'term_handler' TERM INT

echo "Waiting for inference server to be ready on 127.0.0.1:50051 (timeout ${TIMEOUT}s)..."
i=0
until python - <<PY
import socket,sys
s=socket.socket()
try:
 s.settimeout(1)
 s.connect(("127.0.0.1",50051))
 s.close()
 print("ok")
except Exception:
 sys.exit(1)
PY
do
  i=$((i+1))
  if [ "$i" -ge "$TIMEOUT" ]; then
    echo "Inference server did not become ready in ${TIMEOUT}s" >&2
    kill "$SERVER_PID" || true
    exit 1
  fi
  sleep 1
done

echo "Inference server is ready. Starting Streamlit frontend..."
exec /opt/venv/bin/python -u -m streamlit run app/streamlit_app.py --server.address=0.0.0.0 --server.port=8501