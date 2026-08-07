"""
FastAPI evaluation service for scoring RAG (retrieval-augmented generation)
quality against a golden Q&A dataset.

Decoupled from any specific RAG implementation on purpose: you run your
own RAG pipeline (any of the RAG examples in this repo, or your own),
submit what it actually retrieved and answered for each golden question,
and this scores it. That's what makes it reusable as a regression-testing
tool -- point it at any RAG system's outputs, across versions, and compare
runs over time.

Metrics per case:
  - answer_similarity: embedding cosine similarity between the generated
    answer and the golden answer. Deterministic, no LLM call.
  - context_recall: the best embedding similarity between the golden
    answer and any single retrieved chunk -- a proxy for "did retrieval
    actually surface something relevant". Deterministic, no LLM call.
  - faithfulness (LLM-judge): is the generated answer actually supported
    by the retrieved context, or does it say things the context doesn't
    back up (hallucination)?
  - correctness (LLM-judge): does the generated answer correctly answer
    the question, compared to the golden answer?
  - overall_score: the mean of whichever of the above were computed
    successfully for that case.

Faithfulness and correctness are scored together in a single structured
LLM call per case (not two separate calls) to keep judge cost down. If
that call fails for a given case, the case still gets its two
deterministic metrics, is flagged with judge_error, and does not take
down the rest of the batch.

Runs are stored so you can list past runs and compare two of them --
e.g. "did the new chunking strategy actually improve faithfulness, or
just similarity?"

Known limitation, called out rather than hidden: runs are stored in an
in-memory dict, so they don't survive a restart and aren't shared across
multiple worker processes. For a real CI/regression pipeline, persist
runs to a database instead.

Setup:
    pip install fastapi "uvicorn[standard]" pydantic-settings numpy \
        langchain-huggingface langchain-google-genai sentence-transformers \
        python-dotenv

    .env:
        GOOGLE_API_KEY=your_gemini_api_key_here

Run:
    python retrieval_eval_api.py
    # or: uvicorn retrieval_eval_api:app --reload --port 8000

Interactive docs: http://localhost:8000/docs

Example:
    curl -X POST http://localhost:8000/api/evaluate \
        -H "Content-Type: application/json" \
        -d '{
          "cases": [
            {
              "question": "What is your return policy?",
              "golden_answer": "Returns are accepted within 30 days for a full refund.",
              "retrieved_context": ["You can return any item within 30 days of delivery for a full refund."],
              "generated_answer": "You can return items within 30 days for a full refund."
            }
          ]
        }'
"""

import time
import uuid
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
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
    max_cases_per_request: int = Field(200)


try:
    settings = Settings()
except Exception as e:
    raise RuntimeError(f"Missing required configuration. Set GOOGLE_API_KEY (in .env or the environment). Details: {e}")

import os
os.environ["GOOGLE_API_KEY"] = settings.google_api_key


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# --------------------------------------------------------------------------
# LLM-as-judge
# --------------------------------------------------------------------------

class JudgeResult(BaseModel):
    faithfulness: float = Field(..., ge=0, le=1, description="0 = generated answer is unsupported by the retrieved context (hallucinated), 1 = fully grounded in it")
    correctness: float = Field(..., ge=0, le=1, description="0 = generated answer is wrong compared to the golden answer, 1 = fully correct")
    reasoning: str = Field(..., description="One or two sentence explanation of both scores")


JUDGE_PROMPT = """You are evaluating a RAG (retrieval-augmented generation) system's answer.

Question:
{question}

Retrieved context (what the system had available to answer with):
{context}

Golden/reference answer:
{golden_answer}

Generated answer to evaluate:
{generated_answer}

Score two things, each from 0.0 to 1.0:
- faithfulness: is the generated answer supported by the retrieved context, or does it
  state things the context doesn't back up (hallucination)? Score on grounding in the
  context, not on whether it matches the golden answer.
- correctness: does the generated answer correctly answer the question, compared to the
  golden answer? Score on correctness, not on wording similarity.

Give a brief reasoning covering both scores."""


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

resources: Dict[str, object] = {}
runs: Dict[str, "EvalRun"] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    resources["embeddings"] = HuggingFaceEmbeddings(model_name=settings.embedding_model)
    llm = ChatGoogleGenerativeAI(model=settings.llm_model, temperature=0.0)
    resources["judge"] = llm.with_structured_output(JudgeResult)
    yield
    resources.clear()


app = FastAPI(
    title="Retrieval Evaluation API",
    description="Scores a RAG system's actual outputs against a golden Q&A dataset: answer similarity, context recall, and LLM-judged faithfulness/correctness.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class CaseInput(BaseModel):
    question: str = Field(..., min_length=1)
    golden_answer: str = Field(..., min_length=1)
    retrieved_context: List[str] = Field(default_factory=list)
    generated_answer: str = Field(..., min_length=1)


class EvaluateRequest(BaseModel):
    cases: List[CaseInput] = Field(..., min_length=1)
    pass_threshold: float = Field(0.7, ge=0, le=1, description="overall_score at/above this counts as a pass")


class CaseResult(BaseModel):
    question: str
    generated_answer: str
    golden_answer: str
    answer_similarity: float
    context_recall: float
    faithfulness: Optional[float]
    correctness: Optional[float]
    overall_score: float
    judge_reasoning: Optional[str]
    judge_error: Optional[str]
    passed: bool


class EvalRun(BaseModel):
    run_id: str
    created_at: float
    num_cases: int
    pass_threshold: float
    mean_answer_similarity: float
    mean_context_recall: float
    mean_faithfulness: Optional[float]
    mean_correctness: Optional[float]
    mean_overall_score: float
    pass_rate: float
    cases: List[CaseResult]


class RunSummary(BaseModel):
    run_id: str
    created_at: float
    num_cases: int
    mean_overall_score: float
    pass_rate: float


class RunListResponse(BaseModel):
    runs: List[RunSummary]


class CompareResponse(BaseModel):
    run_a: RunSummary
    run_b: RunSummary
    delta_mean_overall_score: float
    delta_pass_rate: float
    delta_mean_answer_similarity: float
    delta_mean_context_recall: float


# --------------------------------------------------------------------------
# Evaluation logic
# --------------------------------------------------------------------------

def evaluate_case(case: CaseInput, threshold: float) -> CaseResult:
    embeddings = resources["embeddings"]
    judge = resources["judge"]

    golden_vec = np.array(embeddings.embed_query(case.golden_answer))
    generated_vec = np.array(embeddings.embed_query(case.generated_answer))
    answer_similarity = cosine_similarity(golden_vec, generated_vec)

    if case.retrieved_context:
        context_vecs = [np.array(v) for v in embeddings.embed_documents(case.retrieved_context)]
        context_recall = max(cosine_similarity(golden_vec, cv) for cv in context_vecs)
    else:
        context_recall = 0.0

    faithfulness: Optional[float] = None
    correctness: Optional[float] = None
    judge_reasoning: Optional[str] = None
    judge_error: Optional[str] = None

    try:
        judged = judge.invoke(JUDGE_PROMPT.format(
            question=case.question,
            context="\n\n".join(case.retrieved_context) or "(no context retrieved)",
            golden_answer=case.golden_answer,
            generated_answer=case.generated_answer,
        ))
        faithfulness = judged.faithfulness
        correctness = judged.correctness
        judge_reasoning = judged.reasoning
    except Exception as e:
        judge_error = str(e)

    components = [answer_similarity, context_recall]
    if faithfulness is not None and correctness is not None:
        components += [faithfulness, correctness]
    overall_score = sum(components) / len(components)

    return CaseResult(
        question=case.question,
        generated_answer=case.generated_answer,
        golden_answer=case.golden_answer,
        answer_similarity=answer_similarity,
        context_recall=context_recall,
        faithfulness=faithfulness,
        correctness=correctness,
        overall_score=overall_score,
        judge_reasoning=judge_reasoning,
        judge_error=judge_error,
        passed=overall_score >= threshold,
    )


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mean_optional(values: List[Optional[float]]) -> Optional[float]:
    present = [v for v in values if v is not None]
    return (sum(present) / len(present)) if present else None


def to_summary(run: EvalRun) -> RunSummary:
    return RunSummary(
        run_id=run.run_id, created_at=run.created_at, num_cases=run.num_cases,
        mean_overall_score=run.mean_overall_score, pass_rate=run.pass_rate,
    )


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/evaluate", response_model=EvalRun)
def evaluate(payload: EvaluateRequest):
    if len(payload.cases) > settings.max_cases_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Too many cases ({len(payload.cases)}); max is {settings.max_cases_per_request} per request.",
        )

    case_results = [evaluate_case(case, payload.pass_threshold) for case in payload.cases]

    run = EvalRun(
        run_id=str(uuid.uuid4()),
        created_at=time.time(),
        num_cases=len(case_results),
        pass_threshold=payload.pass_threshold,
        mean_answer_similarity=_mean([c.answer_similarity for c in case_results]),
        mean_context_recall=_mean([c.context_recall for c in case_results]),
        mean_faithfulness=_mean_optional([c.faithfulness for c in case_results]),
        mean_correctness=_mean_optional([c.correctness for c in case_results]),
        mean_overall_score=_mean([c.overall_score for c in case_results]),
        pass_rate=_mean([1.0 if c.passed else 0.0 for c in case_results]),
        cases=case_results,
    )
    runs[run.run_id] = run
    return run


@app.get("/api/runs", response_model=RunListResponse)
def list_runs(limit: int = Query(20, ge=1, le=200)):
    ordered = sorted(runs.values(), key=lambda r: r.created_at, reverse=True)[:limit]
    return RunListResponse(runs=[to_summary(r) for r in ordered])


@app.get("/api/runs/{run_id}", response_model=EvalRun)
def get_run(run_id: str):
    run = runs.get(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="No run with that ID.")
    return run


@app.get("/api/runs/compare", response_model=CompareResponse)
def compare_runs(run_a: str, run_b: str):
    a, b = runs.get(run_a), runs.get(run_b)
    if not a:
        raise HTTPException(status_code=404, detail=f"No run with id '{run_a}'.")
    if not b:
        raise HTTPException(status_code=404, detail=f"No run with id '{run_b}'.")

    return CompareResponse(
        run_a=to_summary(a),
        run_b=to_summary(b),
        delta_mean_overall_score=b.mean_overall_score - a.mean_overall_score,
        delta_pass_rate=b.pass_rate - a.pass_rate,
        delta_mean_answer_similarity=b.mean_answer_similarity - a.mean_answer_similarity,
        delta_mean_context_recall=b.mean_context_recall - a.mean_context_recall,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
