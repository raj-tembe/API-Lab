"""
FastAPI semantic cache in front of an LLM.

Matches new queries against previously-answered ones by embedding
similarity rather than exact string match, so a paraphrased repeat
question ("what's your refund policy?" vs "how do refunds work?") returns
a cached answer instead of paying for a fresh LLM call.

Self-contained: HuggingFace embeddings for similarity, Gemini for actual
generation on cache misses.

What's implemented, concretely:
  - Cosine-similarity lookup against an in-memory cache, with a
    configurable similarity threshold (default is deliberately high --
    0.95 -- because a false-positive cache hit silently returns the wrong
    answer to the user, which is worse than the cost of a miss).
  - TTL expiration and a max-size cap with LRU eviction, so the cache
    can't grow unbounded or serve indefinitely stale answers.
  - Per-request cache bypass, and stats (hits/misses/hit rate) so you can
    see whether the cache is actually paying for itself.
  - Thread-safe: a lock guards cache reads/writes, since FastAPI runs sync
    endpoints across a thread pool and concurrent requests would otherwise
    race on the same in-memory store.
  - Cache management endpoints (clear all, delete one entry).

Known limitation, called out rather than hidden: the cache is an
in-memory dict, so it does not survive a restart and is not shared across
multiple worker processes. For real multi-worker production use, back
this with Redis (vector sets / RediSearch) or a vector DB instead.
"""

import threading
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = Field(..., description="Gemini API key")
    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    llm_model: str = Field("gemini-2.5-flash")

    similarity_threshold: float = Field(0.95, ge=0.0, le=1.0, description="Minimum cosine similarity to count as a cache hit")
    cache_ttl_seconds: float = Field(3600.0, description="How long a cache entry stays valid")
    cache_max_size: int = Field(1000, description="Max cached entries before LRU eviction kicks in")


try:
    settings = Settings()
except Exception as e:
    raise RuntimeError(f"Missing required configuration. Set GOOGLE_API_KEY (in .env or the environment). Details: {e}")

import os
os.environ["GOOGLE_API_KEY"] = settings.google_api_key


# --------------------------------------------------------------------------
# Semantic cache
# --------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


@dataclass
class CacheEntry:
    id: str
    query: str
    embedding: np.ndarray
    response: str
    created_at: float
    last_accessed: float
    hits: int = 0


class SemanticCache:
    def __init__(self, threshold: float, ttl_seconds: float, max_size: int):
        self.threshold = threshold
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._entries: Dict[str, CacheEntry] = {}
        self._lock = threading.Lock()
        self.hit_count = 0
        self.miss_count = 0

    def _purge_expired_locked(self) -> None:
        now = time.time()
        expired = [eid for eid, e in self._entries.items() if now - e.created_at > self.ttl_seconds]
        for eid in expired:
            del self._entries[eid]

    def lookup(self, query_embedding: np.ndarray) -> Tuple[Optional[CacheEntry], float]:
        with self._lock:
            self._purge_expired_locked()

            best_entry, best_score = None, -1.0
            for entry in self._entries.values():
                score = cosine_similarity(query_embedding, entry.embedding)
                if score > best_score:
                    best_entry, best_score = entry, score

            if best_entry is not None and best_score >= self.threshold:
                best_entry.hits += 1
                best_entry.last_accessed = time.time()
                self.hit_count += 1
                return best_entry, best_score

            self.miss_count += 1
            return None, max(best_score, 0.0)

    def insert(self, query: str, embedding: np.ndarray, response: str) -> str:
        with self._lock:
            self._purge_expired_locked()

            if len(self._entries) >= self.max_size:
                lru_id = min(self._entries, key=lambda eid: self._entries[eid].last_accessed)
                del self._entries[lru_id]

            entry_id = str(uuid.uuid4())
            now = time.time()
            self._entries[entry_id] = CacheEntry(
                id=entry_id, query=query, embedding=embedding, response=response,
                created_at=now, last_accessed=now,
            )
            return entry_id

    def delete(self, entry_id: str) -> bool:
        with self._lock:
            return self._entries.pop(entry_id, None) is not None

    def clear(self) -> int:
        with self._lock:
            count = len(self._entries)
            self._entries.clear()
            return count

    def stats(self) -> dict:
        with self._lock:
            self._purge_expired_locked()
            total = self.hit_count + self.miss_count
            return {
                "total_entries": len(self._entries),
                "hits": self.hit_count,
                "misses": self.miss_count,
                "hit_rate": (self.hit_count / total) if total else 0.0,
            }


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

resources: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources["embeddings"] = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    resources["llm"] = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0.3)
    resources["cache"] = SemanticCache(
        threshold=settings.similarity_threshold,
        ttl_seconds=settings.cache_ttl_seconds,
        max_size=settings.cache_max_size,
    )
    yield
    resources.clear()


app = FastAPI(
    title="Semantic Cache API",
    description="Caches LLM responses by embedding similarity to cut cost and latency on repeated/paraphrased queries.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    bypass_cache: bool = Field(False, description="Skip the cache lookup and force a fresh LLM call (the cache is still updated with the new result)")


class ChatResponse(BaseModel):
    response: str
    cache_hit: bool
    similarity: float
    latency_ms: float
    cache_entry_id: str


class CacheStatsResponse(BaseModel):
    total_entries: int
    hits: int
    misses: int
    hit_rate: float
    similarity_threshold: float
    ttl_seconds: float
    max_size: int


class DeleteResponse(BaseModel):
    deleted: bool


class ClearResponse(BaseModel):
    entries_cleared: int


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    cache: SemanticCache = resources["cache"]
    embeddings = resources["embeddings"]
    llm = resources["llm"]

    start = time.perf_counter()
    query_embedding = np.array(embeddings.embed_query(payload.query))

    if not payload.bypass_cache:
        hit, score = cache.lookup(query_embedding)
        if hit is not None:
            latency_ms = (time.perf_counter() - start) * 1000
            return ChatResponse(
                response=hit.response,
                cache_hit=True,
                similarity=score,
                latency_ms=latency_ms,
                cache_entry_id=hit.id,
            )

    try:
        result = llm.invoke(payload.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")

    entry_id = cache.insert(payload.query, query_embedding, result.content)
    latency_ms = (time.perf_counter() - start) * 1000

    return ChatResponse(
        response=result.content,
        cache_hit=False,
        similarity=0.0,
        latency_ms=latency_ms,
        cache_entry_id=entry_id,
    )


@app.get("/api/cache/stats", response_model=CacheStatsResponse)
def cache_stats():
    cache: SemanticCache = resources["cache"]
    return CacheStatsResponse(
        **cache.stats(),
        similarity_threshold=cache.threshold,
        ttl_seconds=cache.ttl_seconds,
        max_size=cache.max_size,
    )


@app.delete("/api/cache/{entry_id}", response_model=DeleteResponse)
def delete_entry(entry_id: str):
    cache: SemanticCache = resources["cache"]
    deleted = cache.delete(entry_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No cache entry with that ID.")
    return DeleteResponse(deleted=True)


@app.delete("/api/cache", response_model=ClearResponse)
def clear_cache():
    cache: SemanticCache = resources["cache"]
    return ClearResponse(entries_cleared=cache.clear())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
