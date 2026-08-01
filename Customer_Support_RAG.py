#FastAPI + LangChain RAG agent that answers customer support queries.

import os
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from sentence_transformers import CrossEncoder

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Create a .env file with "
        "GOOGLE_API_KEY=your_api_key_here and restart the app."
    )
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

PERSIST_DIR = "chroma_customer_support"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LLM_MODEL = "gemini-2.5-flash"

# A tiny starter knowledge base so /api/chat is testable immediately.
# Add your own support docs for real use via POST /api/ingest.
SEED_DOCS = [
    Document(
        page_content=(
            "Standard shipping takes 3-5 business days and is free on orders "
            "over $50. Express shipping takes 1-2 business days and costs an "
            "additional $9.99."
        ),
        metadata={"source": "shipping-policy"},
    ),
    Document(
        page_content=(
            "You can return any item within 30 days of delivery for a full "
            "refund, as long as it is unused and in its original packaging. "
            "Refunds are issued to the original payment method within 5-7 "
            "business days of us receiving the return."
        ),
        metadata={"source": "return-policy"},
    ),
    Document(
        page_content=(
            "To reset your password, go to the login page and click 'Forgot "
            "password'. You'll receive a reset link by email within a few "
            "minutes. If it doesn't arrive, check your spam folder before "
            "contacting support."
        ),
        metadata={"source": "account-help"},
    ),
    Document(
        page_content=(
            "Our support team is available Monday-Friday, 9am-6pm EST, via "
            "chat and email at support@example.com. Average response time is "
            "under 2 hours during business hours."
        ),
        metadata={"source": "support-hours"},
    ),
]

CUSTOMER_SUPPORT_PROMPT = ChatPromptTemplate.from_template(
    """You are a friendly, professional customer support agent.
Answer the customer's question using ONLY the context below.

Rules:
- If the answer isn't in the context, say you don't have that information
  and offer to connect them with a human support agent. Never guess or
  invent policy details.
- Keep answers short, clear, and warm.
- If the customer sounds frustrated or upset, briefly acknowledge that
  before answering.

Context:
{context}

Customer question:
{question}

Answer:"""
)

# Populated at startup by the lifespan handler below; request handlers read
# from this instead of touching module-level globals directly, so it's
# obvious everything heavy is loaded once, not per-request.
resources: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma(
        collection_name="customer_support",
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # Seed with the starter FAQ only if the collection is empty, so restarts
    # don't keep re-adding duplicate chunks.
    if vectorstore._collection.count() == 0:
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        vectorstore.add_documents(splitter.split_documents(SEED_DOCS))

    resources["vectorstore"] = vectorstore
    resources["reranker"] = CrossEncoder(RERANKER_MODEL)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.3)
    resources["chain"] = CUSTOMER_SUPPORT_PROMPT | llm

    yield
    resources.clear()


app = FastAPI(
    title="Customer Support RAG API",
    description="A LangChain-powered RAG agent that answers customer queries grounded in your knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The customer's question")
    top_k: int = Field(3, ge=1, le=10, description="Number of source chunks to ground the answer in")


class Source(BaseModel):
    content: str
    metadata: dict


class ChatResponse(BaseModel):
    response: str
    sources: List[Source]


class IngestDocument(BaseModel):
    text: str = Field(..., min_length=1)
    source: Optional[str] = Field(None, description="A label for where this content came from")


class IngestRequest(BaseModel):
    documents: List[IngestDocument]


class IngestResponse(BaseModel):
    chunks_added: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    vectorstore = resources["vectorstore"]
    reranker = resources["reranker"]
    chain = resources["chain"]

    query = payload.query.strip()

    # Retrieve a broad candidate set, then rerank down to the most relevant
    # chunks before they reach the LLM.
    candidates = vectorstore.similarity_search(query, k=max(payload.top_k * 3, 10))

    if not candidates:
        return ChatResponse(
            response=(
                "I don't have any information on that yet. Let me connect "
                "you with a human support agent who can help."
            ),
            sources=[],
        )

    pairs = [[query, doc.page_content] for doc in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
    top_docs = [doc for doc, _ in ranked[: payload.top_k]]

    context = "\n\n".join(doc.page_content for doc in top_docs)

    try:
        result = chain.invoke({"context": context, "question": query})
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {e}")

    return ChatResponse(
        response=result.content,
        sources=[Source(content=doc.page_content, metadata=doc.metadata) for doc in top_docs],
    )


@app.post("/api/ingest", response_model=IngestResponse)
def ingest(payload: IngestRequest):
    if not payload.documents:
        raise HTTPException(status_code=400, detail="At least one document is required.")

    vectorstore = resources["vectorstore"]
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    docs = [
        Document(page_content=d.text, metadata={"source": d.source or "api-ingest"})
        for d in payload.documents
    ]
    chunks = splitter.split_documents(docs)

    if not chunks:
        raise HTTPException(status_code=400, detail="Documents contained no usable text.")

    vectorstore.add_documents(chunks)

    return IngestResponse(chunks_added=len(chunks))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
