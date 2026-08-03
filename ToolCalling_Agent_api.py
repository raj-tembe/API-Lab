"""
FastAPI + LangGraph tool-calling agent with per-session conversation memory
and token streaming.

None of the other FastAPI examples in this repo give the model tools to
call, and all of them are single-turn (no conversation memory across
requests). This one fills both gaps: it's a ReAct-style agent
(`langgraph.prebuilt.create_react_agent`) that can decide on its own to
call a calculator, a unit converter, or a clock tool, and it remembers
each session's conversation via LangGraph checkpointing, keyed by a
`session_id` you choose.

Self-contained and free to run: the tools need no external API keys or
network calls, so the only requirement is GOOGLE_API_KEY for the LLM
itself.
"""

import ast
import json
import operator
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not GOOGLE_API_KEY:
    raise RuntimeError(
        "GOOGLE_API_KEY is not set. Create a .env file with "
        "GOOGLE_API_KEY=your_api_key_here and restart the app."
    )
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

LLM_MODEL = "gemini-2.5-flash"

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a calculator, a unit "
    "converter, and the current date/time. Use a tool whenever it would "
    "give a more accurate answer than reasoning alone -- especially for "
    "arithmetic. Keep answers concise."
)


# --------------------------------------------------------------------------
# Tools -- all self-contained, no external API keys or network calls.
# --------------------------------------------------------------------------

_SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST):
    # Deliberately not using eval()/exec(): this walks a parsed expression
    # tree and only permits numeric literals and arithmetic operators, so
    # there's no way for a crafted "expression" to run arbitrary code.
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPERATORS:
        return _SAFE_OPERATORS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression syntax near: {ast.dump(node)}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression, e.g. '18/100 * 245' or '(3 + 4) ** 2'.
    Supports + - * / % ** and parentheses. Not a general code interpreter."""
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


_LENGTH_TO_METERS = {"m": 1.0, "km": 1000.0, "mi": 1609.344, "ft": 0.3048, "cm": 0.01}
_WEIGHT_TO_KG = {"kg": 1.0, "g": 0.001, "lb": 0.45359237}


@tool
def convert_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert a numeric value between common units.
    Length: m, km, mi, ft, cm. Weight: kg, g, lb. Temperature: c, f."""
    from_unit, to_unit = from_unit.lower(), to_unit.lower()

    if from_unit in ("c", "f") or to_unit in ("c", "f"):
        if from_unit == to_unit:
            return str(value)
        if from_unit == "c" and to_unit == "f":
            return str(value * 9 / 5 + 32)
        if from_unit == "f" and to_unit == "c":
            return str((value - 32) * 5 / 9)
        return f"Error: can't convert temperature unit '{from_unit}' to '{to_unit}'"

    for table, kind in ((_LENGTH_TO_METERS, "length"), (_WEIGHT_TO_KG, "weight")):
        if from_unit in table and to_unit in table:
            base = value * table[from_unit]
            return str(base / table[to_unit])

    return f"Error: unsupported unit pair '{from_unit}' -> '{to_unit}'"


@tool
def get_current_datetime() -> str:
    """Get the current UTC date and time."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


TOOLS = [calculator, convert_units, get_current_datetime]


# --------------------------------------------------------------------------
# Startup
# --------------------------------------------------------------------------

resources: Dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2)
    checkpointer = MemorySaver()
    resources["checkpointer"] = checkpointer
    resources["agent"] = create_react_agent(llm, TOOLS, prompt=SYSTEM_PROMPT, checkpointer=checkpointer)
    yield
    resources.clear()


app = FastAPI(
    title="Tool-Calling Agent API",
    description="A ReAct agent (calculator, unit converter, clock) with per-session memory and streaming.",
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# Schemas
# --------------------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)


class ToolCallInfo(BaseModel):
    tool: str
    input: dict
    output: str


class ChatResponse(BaseModel):
    session_id: str
    response: str
    tool_calls: List[ToolCallInfo]


class HistoryMessage(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    messages: List[HistoryMessage]


class DeleteResponse(BaseModel):
    session_id: str
    deleted: bool


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _config_for(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


def _extract_tool_calls(new_messages: list) -> List[ToolCallInfo]:
    calls_by_id = {}
    for msg in new_messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                calls_by_id[tc["id"]] = {"tool": tc["name"], "input": tc["args"], "output": ""}
        elif isinstance(msg, ToolMessage) and msg.tool_call_id in calls_by_id:
            calls_by_id[msg.tool_call_id]["output"] = str(msg.content)
    return [ToolCallInfo(**c) for c in calls_by_id.values()]


def _role_for(msg) -> str:
    if isinstance(msg, HumanMessage):
        return "user"
    if isinstance(msg, ToolMessage):
        return "tool"
    return "assistant"


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat/{session_id}", response_model=ChatResponse)
def chat(session_id: str, payload: ChatRequest):
    agent = resources["agent"]
    config = _config_for(session_id)

    prior_state = agent.get_state(config)
    prior_count = len(prior_state.values.get("messages", [])) if prior_state.values else 0

    try:
        result = agent.invoke({"messages": [HumanMessage(content=payload.message)]}, config=config)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent invocation failed: {e}")

    all_messages = result["messages"]
    new_messages = all_messages[prior_count:]

    return ChatResponse(
        session_id=session_id,
        response=all_messages[-1].content,
        tool_calls=_extract_tool_calls(new_messages),
    )


@app.post("/api/chat/{session_id}/stream")
async def chat_stream(session_id: str, payload: ChatRequest):
    agent = resources["agent"]
    config = _config_for(session_id)

    async def event_generator():
        try:
            async for event in agent.astream_events(
                {"messages": [HumanMessage(content=payload.message)]}, config=config, version="v2"
            ):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        yield f"event: token\ndata: {json.dumps({'content': chunk.content})}\n\n"

                elif kind == "on_tool_start":
                    payload_data = {"tool": event["name"], "input": event["data"].get("input", {})}
                    yield f"event: tool_call\ndata: {json.dumps(payload_data)}\n\n"

                elif kind == "on_tool_end":
                    payload_data = {"tool": event["name"], "output": str(event["data"].get("output", ""))}
                    yield f"event: tool_result\ndata: {json.dumps(payload_data)}\n\n"

            yield f"event: done\ndata: {json.dumps({'session_id': session_id})}\n\n"

        except Exception as e:
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/chat/{session_id}/history", response_model=HistoryResponse)
def history(session_id: str):
    agent = resources["agent"]
    state = agent.get_state(_config_for(session_id))
    messages = state.values.get("messages", []) if state.values else []

    return HistoryResponse(
        session_id=session_id,
        messages=[
            HistoryMessage(role=_role_for(m), content=str(m.content))
            for m in messages
            if str(m.content)
        ],
    )


@app.delete("/api/chat/{session_id}", response_model=DeleteResponse)
def delete_session(session_id: str):
    checkpointer = resources["checkpointer"]
    checkpointer.delete_thread(session_id)
    return DeleteResponse(session_id=session_id, deleted=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
