#!/usr/bin/env bash
if [ "$SERVICE" = "ui" ]; then
  streamlit run src/ui/app.py --server.port $PORT
else
  uvicorn src.api.main:app --host 0.0.0.0 --port $PORT
fi
