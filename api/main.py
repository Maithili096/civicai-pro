"""
api/main.py
────────────
FastAPI application for CivicAI-Pro.

FIX: Previously the endpoint called run_graph() directly and let
     unhandled LangGraph exceptions propagate as 500s with no body.
     Now all errors are caught and returned as structured JSON.
"""
from __future__ import annotations
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

# ─── Lazy import so startup errors are clear ─────────────────────────────────
_graph = None

def get_graph():
    global _graph
    if _graph is None:
        from agents.graph import civicai_graph
        _graph = civicai_graph
    return _graph


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    query:       str
    answer:      str
    language:    str
    agent_used:  str
    sources:     list[str]
    confidence:  float
    error:       str


class IngestRequest(BaseModel):
    texts:     list[str] = Field(..., min_items=1)
    metadatas: list[dict[str, Any]] | None = None


# ─── App ──────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-warm graph on startup (catches config errors early)
    try:
        get_graph()
        print("[CivicAI-Pro] Graph compiled and ready.")
    except Exception as exc:
        print(f"[CivicAI-Pro] WARNING: Graph failed to pre-warm: {exc}")
    yield


app = FastAPI(
    title="CivicAI-Pro",
    description="Multilingual Agentic AI for Civic Intelligence",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest):
    """
    Main query endpoint. Runs full LangGraph pipeline.
    Always returns a structured response — never a naked 500.
    """
    from agents.graph import run_graph
    try:
        state = run_graph(req.query, req.metadata)
        return QueryResponse(
            query      = req.query,
            answer     = state.get("final_answer", ""),
            language   = state.get("language", "en"),
            agent_used = state.get("agent_used", "unknown"),
            sources    = state.get("sources", []),
            confidence = state.get("confidence", 0.0),
            error      = state.get("error", ""),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/ingest")
async def ingest_documents(req: IngestRequest):
    """Ingest raw texts into the FAISS vector store."""
    from rag.pipeline import build_index
    try:
        store = build_index(req.texts, req.metadatas)
        return {"status": "ok", "chunks_indexed": store.index.ntotal}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/agents")
async def list_agents():
    return {
        "agents": [
            {"name": "rag_agent",   "intent": "rag",       "description": "Retrieval-augmented generation"},
            {"name": "nlp_agent",   "intent": "nlp",       "description": "Classification, translation, NER"},
            {"name": "summarizer",  "intent": "summarize", "description": "Document summarization"},
            {"name": "civic_qa",    "intent": "civic_qa",  "description": "Policy & government scheme QA"},
        ]
    }
