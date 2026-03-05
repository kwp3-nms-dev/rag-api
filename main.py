import logging
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel

from config import ALLOWED_COLLECTIONS, settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("rag-api")

# Runtime allowlist — starts from config, grows via /admin/collections/register
_allowed_collections: set = set(ALLOWED_COLLECTIONS)

# --- Auth ---

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)


async def require_api_key(key: str = Security(api_key_header)) -> None:
    if key != settings.API_KEY:
        log.warning("Rejected request with invalid API key")
        raise HTTPException(status_code=403, detail="Invalid API key")


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.client = httpx.AsyncClient()
    yield
    await app.state.client.aclose()


app = FastAPI(title="RAG API", lifespan=lifespan)


# --- Request models ---

class IngestRequest(BaseModel):
    collection: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class QueryRequest(BaseModel):
    collection: str
    query: str
    top_k: Optional[int] = None


# --- Validation ---

def validate_collection(collection: str) -> None:
    if collection not in _allowed_collections:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown collection '{collection}'. Allowed: {sorted(_allowed_collections)}",
        )


# --- Ollama helpers ---

async def embed_text(text: str) -> List[float]:
    client: httpx.AsyncClient = app.state.client
    try:
        resp = await client.post(
            f"{settings.OLLAMA_URL}/api/embeddings",
            json={"model": settings.EMBED_MODEL, "prompt": text, "keep_alive": 0},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception as e:
        log.error("Ollama embedding error: %s", e)
        raise HTTPException(status_code=502, detail="Embedding service unavailable")


async def llm_generate(prompt: str) -> str:
    client: httpx.AsyncClient = app.state.client
    try:
        resp = await client.post(
            f"{settings.OLLAMA_URL}/api/generate",
            json={"model": settings.LLM_MODEL, "prompt": prompt, "stream": False, "keep_alive": -1, "options": {"num_ctx": settings.LLM_NUM_CTX}},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()["response"]
    except Exception as e:
        log.error("Ollama generate error: %s", e)
        raise HTTPException(status_code=502, detail="LLM service unavailable")


# --- Qdrant helpers ---

async def qdrant_search(collection: str, vector: List[float], top_k: int) -> List[Dict]:
    client: httpx.AsyncClient = app.state.client
    try:
        resp = await client.post(
            f"{settings.QDRANT_URL}/collections/{collection}/points/search",
            json={"vector": vector, "limit": top_k, "with_payload": True},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as e:
        log.error("Qdrant search error: %s", e)
        raise HTTPException(status_code=502, detail="Vector search unavailable")


async def qdrant_upsert(collection: str, point_id: str, vector: List[float], payload: Dict) -> None:
    client: httpx.AsyncClient = app.state.client
    try:
        resp = await client.put(
            f"{settings.QDRANT_URL}/collections/{collection}/points",
            json={"points": [{"id": point_id, "vector": vector, "payload": payload}]},
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        log.error("Qdrant upsert error: %s", e)
        raise HTTPException(status_code=502, detail="Vector store unavailable")


# --- Request models ---

class RegisterRequest(BaseModel):
    name: str
    vector_size: int = 2560


# --- Endpoints ---

@app.post("/admin/collections/register", dependencies=[Security(require_api_key)])
async def register_collection(req: RegisterRequest):
    name = req.name.strip()
    if not name or not name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Invalid collection name. Use alphanumeric + underscores only.")

    client: httpx.AsyncClient = app.state.client

    # Check if already exists in Qdrant
    r = await client.get(f"{settings.QDRANT_URL}/collections/{name}", timeout=5)
    if r.status_code == 200:
        _allowed_collections.add(name)
        log.info("Collection '%s' already exists — added to allowlist", name)
        return {"collection": name, "status": "already_exists"}

    # Create in Qdrant
    try:
        r = await client.put(
            f"{settings.QDRANT_URL}/collections/{name}",
            json={"vectors": {"size": req.vector_size, "distance": "Cosine"}},
            timeout=10,
        )
        r.raise_for_status()
    except Exception as e:
        log.error("Failed to create Qdrant collection '%s': %s", name, e)
        raise HTTPException(status_code=502, detail="Failed to create collection in Qdrant")

    _allowed_collections.add(name)
    log.info("Registered new collection '%s' (vector_size=%d)", name, req.vector_size)
    return {"collection": name, "status": "created"}


@app.get("/health")
async def health():
    client: httpx.AsyncClient = app.state.client
    qdrant_ok, ollama_ok = False, False
    try:
        r = await client.get(f"{settings.QDRANT_URL}/healthz", timeout=5)
        qdrant_ok = r.status_code == 200
    except Exception:
        pass
    try:
        r = await client.get(f"{settings.OLLAMA_URL}/api/tags", timeout=5)
        ollama_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "qdrant": "ok" if qdrant_ok else "unavailable",
        "ollama": "ok" if ollama_ok else "unavailable",
    }


@app.post("/ingest", dependencies=[Security(require_api_key)])
async def ingest(req: IngestRequest):
    validate_collection(req.collection)
    embedding = await embed_text(req.text)
    point_id = str(uuid.uuid4())
    payload = {**(req.metadata or {}), "text": req.text}
    await qdrant_upsert(req.collection, point_id, embedding, payload)
    log.info("Ingested point %s into %s", point_id, req.collection)
    return {"id": point_id, "collection": req.collection, "status": "ok"}


@app.post("/query", dependencies=[Security(require_api_key)])
async def query(req: QueryRequest):
    validate_collection(req.collection)
    top_k = req.top_k if req.top_k is not None else settings.TOP_K
    embedding = await embed_text(req.query)
    hits = await qdrant_search(req.collection, embedding, top_k)

    def _trim(text: str, max_words: int) -> str:
        words = text.split()
        return " ".join(words[:max_words]) + ("…" if len(words) > max_words else "")

    sources = [
        _trim(h["payload"].get("text", ""), settings.SOURCE_MAX_WORDS)
        for h in hits if h.get("payload", {}).get("text")
    ]

    if not sources:
        log.info("Query on %s returned no sources: %r", req.collection, req.query)
        return {"answer": "No relevant sources found.", "sources": []}

    prompt = (
        "You are a helpful IT assistant. Answer the question using only the sources below.\n\n"
        + "\n---\n".join(f"Source {i+1}:\n{s}" for i, s in enumerate(sources))
        + f"\n\nQuestion: {req.query}\nAnswer:"
    )
    answer = await llm_generate(prompt)
    log.info("Query on %s returned %d sources", req.collection, len(sources))
    return {"answer": answer, "sources": sources}


@app.get("/collections", dependencies=[Security(require_api_key)])
async def list_collections():
    client: httpx.AsyncClient = app.state.client
    try:
        resp = await client.get(f"{settings.QDRANT_URL}/collections", timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log.error("Qdrant collections error: %s", e)
        raise HTTPException(status_code=502, detail="Vector store unavailable")

    collections = []
    for col in resp.json().get("result", {}).get("collections", []):
        name = col.get("name")
        count = 0
        try:
            r2 = await client.post(
                f"{settings.QDRANT_URL}/collections/{name}/points/count",
                json={},
                timeout=5,
            )
            if r2.status_code == 200:
                count = r2.json().get("result", {}).get("count", 0)
        except Exception:
            pass
        collections.append({"name": name, "points": count})
    return {"collections": collections}
