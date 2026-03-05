# RAG API

A lightweight FastAPI orchestration layer for a local RAG (Retrieval-Augmented Generation) pipeline using Qdrant and Ollama.

## Stack
- **Vector store:** Qdrant
- **Embeddings:** Ollama (default: `qwen3-embedding:4b`)
- **LLM:** Ollama (default: `qwen3.5:9b`)
- **Framework:** FastAPI (fully async)

## Requirements
- Qdrant running at `localhost:6333`
- Ollama running at `localhost:11434` with your chosen models pulled

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configure

```bash
cp .env.example .env
# Edit .env — set API_KEY and adjust URLs/models as needed
```

## Run

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

## Endpoints

### Health check
```bash
curl http://localhost:8000/health
```

### Ingest a document
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"collection": "my_collection", "text": "Your document text here.", "metadata": {"source": "manual"}}'
```

### Query
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{"collection": "my_collection", "query": "Your question here?", "top_k": 5}'
```

### List collections
```bash
curl -H "X-API-Key: your-key" http://localhost:8000/collections
```

## Security
- All endpoints except `/health` require an `X-API-Key` header
- Collection names are validated against a configurable allowlist (`config.py`)
- API key is set via `API_KEY` in `.env` — generate one with:
  ```bash
  python3 -c "import secrets; print(secrets.token_hex(32))"
  ```

## Collections
Define your collections in `config.py` under `ALLOWED_COLLECTIONS`. Create them in Qdrant before ingesting — vector size must match your embedding model output (e.g. `2560` for `qwen3-embedding:4b`).
