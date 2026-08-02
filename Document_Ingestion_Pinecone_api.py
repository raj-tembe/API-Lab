"""
Production-grade FastAPI service for document ingestion into Pinecone.

Self-contained: embeddings, chunking, Pinecone index management, and
ingestion are all implemented in this file. Kept as a single file to match
this repo's one-file-per-example convention -- in a larger production
deployment you'd typically split Settings/routers/services into separate
modules.

What "production-grade" means here, concretely:
  - Config via environment variables (pydantic-settings), validated at
    startup instead of failing deep inside a request.
  - API-key auth on every mutating endpoint.
  - Ingestion runs as a background job (returns a job_id immediately)
    instead of blocking the HTTP request for however long embedding +
    upserting a large document takes.
  - Idempotent upserts: chunk IDs are a hash of (source, chunk index,
    content), so re-ingesting the same file overwrites the same vectors
    instead of duplicating them.
  - Batched, threaded upserts (Pinecone SDK's own batching), file size/type
    validation, and a /health check that verifies real Pinecone
    connectivity rather than just "the process is running".
"""

import hashlib
import logging
import os
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from enum import Enum
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from pinecone import Pinecone, ServerlessSpec

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("document_ingestion")


# Settings 

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    pinecone_api_key: str = Field(..., description="Pinecone API key")
    pinecone_index_name: str = Field("document-ingestion", description="Pinecone index to use/create")
    pinecone_cloud: str = Field("aws", description="Serverless cloud provider for index creation")
    pinecone_region: str = Field("us-east-1", description="Serverless region for index creation")

    ingestion_api_key: str = Field(..., description="Required value of the X-API-Key header on mutating endpoints")

    embedding_model: str = Field("sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = Field(384, description="Must match the embedding model's output size")

    chunk_size: int = Field(1000)
    chunk_overlap: int = Field(200)

    max_file_size_mb: int = Field(20)
    allowed_extensions: tuple = (".pdf", ".txt", ".md", ".csv")


try:
    settings = Settings()
except Exception as e:
    raise RuntimeError(
        "Missing required configuration. Set PINECONE_API_KEY and "
        f"INGESTION_API_KEY (in a .env file or the environment). Details: {e}"
    )


# Auth 

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(api_key: Optional[str] = Security(_api_key_header)) -> None:
    if not api_key or api_key != settings.ingestion_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid X-API-Key header")


# Job tracking (in-memory -- see the module docstring's limitation note)

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(BaseModel):
    job_id: str
    status: JobStatus
    source: str
    namespace: str
    chunks_ingested: int = 0
    error: Optional[str] = None
    created_at: float
    updated_at: float


jobs: Dict[str, Job] = {}

# Heavy, shared resources -- populated once at startup by lifespan(), not
# per-request.
resources: Dict[str, object] = {}


# Startup / shutdown 

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading embedding model %s", settings.embedding_model)
    resources["embeddings"] = HuggingFaceEmbeddings(model_name=settings.embedding_model)

    pc = Pinecone(api_key=settings.pinecone_api_key)
    resources["pinecone_client"] = pc

    if not pc.has_index(settings.pinecone_index_name):
        logger.info("Creating Pinecone index %s (dim=%s)", settings.pinecone_index_name, settings.embedding_dimension)
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )

    resources["index"] = pc.Index(settings.pinecone_index_name)
    logger.info("Ready. Connected to Pinecone index %s", settings.pinecone_index_name)

    yield
    resources.clear()


app = FastAPI(
    title="Document Ingestion API",
    description="Production-grade document ingestion into Pinecone: chunking, embedding, idempotent upserts, and async job tracking.",
    version="1.0.0",
    lifespan=lifespan,
)

# Schemas

class IngestTextRequest(BaseModel):
    text: str = Field(..., min_length=1)
    source: str = Field(..., min_length=1, description="A label identifying where this content came from, e.g. a filename or URL")
    namespace: str = Field("", description="Pinecone namespace to isolate this content in (e.g. per tenant)")


class IngestResponse(BaseModel):
    job_id: str
    status: JobStatus


class DeleteResponse(BaseModel):
    source: str
    namespace: str
    deleted: bool


class HealthResponse(BaseModel):
    status: str
    index_name: str
    total_vector_count: int
    namespaces: List[str]

# Core ingestion logic (shared by both entry points, runs as a background task)

def chunk_ids_for(source: str, namespace: str, chunks: List[Document]) -> List[str]:
    """Deterministic IDs so re-ingesting identical content overwrites the
    same vectors instead of duplicating them."""
    ids = []
    for i, chunk in enumerate(chunks):
        digest = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()[:16]
        ids.append(f"{namespace or 'default'}:{source}:{i}:{digest}")
    return ids


def run_ingestion_job(job_id: str, documents: List[Document], source: str, namespace: str) -> None:
    job = jobs[job_id]
    job.status = JobStatus.PROCESSING
    job.updated_at = time.time()

    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        chunks = splitter.split_documents(documents)

        if not chunks:
            raise ValueError("Document contained no usable text after chunking.")

        embeddings = resources["embeddings"]
        index = resources["index"]

        vectors = embeddings.embed_documents([c.page_content for c in chunks])
        ids = chunk_ids_for(source, namespace, chunks)

        records = [
            {
                "id": vec_id,
                "values": vector,
                "metadata": {"source": source, "text": chunk.page_content, **chunk.metadata},
            }
            for vec_id, vector, chunk in zip(ids, vectors, chunks)
        ]

        index.upsert(vectors=records, namespace=namespace, batch_size=100)

        job.status = JobStatus.COMPLETED
        job.chunks_ingested = len(chunks)
        logger.info("Job %s completed: %d chunks ingested for source=%s", job_id, len(chunks), source)

    except Exception as e:
        job.status = JobStatus.FAILED
        job.error = str(e)
        logger.exception("Job %s failed for source=%s", job_id, source)

    finally:
        job.updated_at = time.time()


def new_job(source: str, namespace: str) -> Job:
    job_id = str(uuid.uuid4())
    now = time.time()
    job = Job(job_id=job_id, status=JobStatus.PENDING, source=source, namespace=namespace, created_at=now, updated_at=now)
    jobs[job_id] = job
    return job


# Endpoints 

@app.get("/health", response_model=HealthResponse)
def health():
    try:
        stats = resources["index"].describe_index_stats()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Pinecone is unreachable: {e}")

    return HealthResponse(
        status="ok",
        index_name=settings.pinecone_index_name,
        total_vector_count=stats.total_vector_count,
        namespaces=list(stats.namespaces.keys()) if stats.namespaces else [],
    )


@app.post(
    "/api/documents/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
)
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    namespace: str = "",
):
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(settings.allowed_extensions)}",
        )

    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_file_size_mb:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File is {size_mb:.1f}MB, exceeds the {settings.max_file_size_mb}MB limit.",
        )

    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    if ext == ".pdf":
        # PyPDFLoader needs a real file path, so write the upload to a temp file.
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            documents = PyPDFLoader(tmp_path).load()
        finally:
            os.unlink(tmp_path)
        for d in documents:
            d.metadata["source"] = filename
    else:
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="File is not valid UTF-8 text.")
        documents = [Document(page_content=text, metadata={"source": filename})]

    job = new_job(source=filename, namespace=namespace)
    background_tasks.add_task(run_ingestion_job, job.job_id, documents, filename, namespace)

    return IngestResponse(job_id=job.job_id, status=job.status)


@app.post(
    "/api/documents/ingest-text",
    response_model=IngestResponse,
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(require_api_key)],
)
async def ingest_text(payload: IngestTextRequest, background_tasks: BackgroundTasks):
    documents = [Document(page_content=payload.text, metadata={"source": payload.source})]

    job = new_job(source=payload.source, namespace=payload.namespace)
    background_tasks.add_task(run_ingestion_job, job.job_id, documents, payload.source, payload.namespace)

    return IngestResponse(job_id=job.job_id, status=job.status)


@app.get(
    "/api/documents/jobs/{job_id}",
    response_model=Job,
    dependencies=[Depends(require_api_key)],
)
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No job with that ID.")
    return job


@app.delete(
    "/api/documents/by-source/{source}",
    response_model=DeleteResponse,
    dependencies=[Depends(require_api_key)],
)
def delete_by_source(source: str, namespace: str = ""):
    index = resources["index"]
    try:
        index.delete(filter={"source": {"$eq": source}}, namespace=namespace)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Pinecone delete failed: {e}")

    return DeleteResponse(source=source, namespace=namespace, deleted=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
