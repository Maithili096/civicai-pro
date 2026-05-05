"""
core/state.py
─────────────
FIX: All LangGraph nodes MUST return dict[str, Any].
     Never return None, a string, or a bare object.
     This file defines the single shared StateGraph schema.
"""
from __future__ import annotations
from typing import Any
from typing_extensions import TypedDict


class AgentState(TypedDict, total=False):
    """Shared state passed between every LangGraph node."""
    query: str                  # Original user query
    language: str               # Detected language code (e.g. 'en', 'hi', 'mr')
    intent: str                 # Classified intent (rag | nlp | summarize | civic_qa)
    context_docs: list[str]     # Retrieved RAG chunks
    agent_answer: str           # Raw answer from the specialist agent
    final_answer: str           # Post-processed final answer
    sources: list[str]          # Source document names
    agent_used: str             # Which agent handled the query
    confidence: float           # 0.0 – 1.0
    error: str                  # Any error message (empty string = no error)
    metadata: dict[str, Any]    # Passthrough metadata
