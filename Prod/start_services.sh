#!/bin/bash
set -e

docker compose up -d postgres_db
docker compose run --rm ingest_pg

docker compose up -d redis
docker compose up -d chroma_db
docker compose run --rm ingest_chroma

docker compose up -d ollama
docker compose run --rm ollama-model-init

docker compose up -d als_trainer
docker compose up -d als_reco_engine
docker compose up -d chat_reco_engine
