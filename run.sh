#!/bin/sh
# PandaRAG — start the app on http://localhost:8787
# Optional: ANTHROPIC_API_KEY for generative answers, PANDARAG_EMBED_MODEL for
# sentence-transformers embeddings, PANDARAG_NCBI_EMAIL for PubMed etiquette.
cd "$(dirname "$0")/backend" || exit 1
exec uvicorn app:app --port "${PORT:-8787}" --host 127.0.0.1
