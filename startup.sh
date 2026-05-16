#!/bin/bash
if [ ! -d "chroma_db" ]; then
    echo "ChromaDB not found — running ingestion..."
    python ingest.py
    echo "Ingestion complete"
fi
uvicorn main:app --host 0.0.0.0 --port $PORT
