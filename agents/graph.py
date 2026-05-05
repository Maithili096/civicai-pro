"""
agents/graph.py
────────────────
FIX: run_graph() was failing with 500 because:
  1. Nodes returned non-dict values
  2. Conditional edges used string keys that didn't match node names
  3. Graph was never compiled before .invoke()

This file builds and compiles the StateGraph correctly.
"""


from __future__ import annotations
from typing import Any

from langgraph.graph import StateGraph, END

from core.state import AgentState
from agents.nodes import (
    router_node,
    rag_agent_node,
    nlp_agent_node,
    summarizer_node,
    civic_qa_node,
    aggregator_node,
)


def _route_by_intent(state: AgentState) -> str:
    """
    Conditional edge function.
    MUST return a string that exactly matches a node name added to the graph.
    """
    intent = state.get("intent", "rag")
    mapping = {
        "rag":        "rag_agent",
        "nlp":        "nlp_agent",
        "summarize":  "summarizer",
        "civic_qa":   "civic_qa",
    }
    return mapping.get(intent, "rag_agent")   # safe fallback


def build_graph():
    """
    Builds, wires, and compiles the CivicAI-Pro LangGraph.

    Flow:
      router → [conditional] → specialist_agent → aggregator → END
    """
    graph = StateGraph(AgentState)

    # 1. Register nodes (name must match what _route_by_intent returns)
    graph.add_node("router",     router_node)
    graph.add_node("rag_agent",  rag_agent_node)
    graph.add_node("nlp_agent",  nlp_agent_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("civic_qa",   civic_qa_node)
    graph.add_node("aggregator", aggregator_node)

    # 2. Entry point
    graph.set_entry_point("router")

    # 3. Conditional routing after router
    graph.add_conditional_edges(
        "router",
        _route_by_intent,
        {
            "rag_agent":  "rag_agent",
            "nlp_agent":  "nlp_agent",
            "summarizer": "summarizer",
            "civic_qa":   "civic_qa",
        },
    )

    # 4. All agents converge to aggregator
    for agent in ("rag_agent", "nlp_agent", "summarizer", "civic_qa"):
        graph.add_edge(agent, "aggregator")

    # 5. Aggregator → END
    graph.add_edge("aggregator", END)

    # 6. Compile (REQUIRED — calling .invoke() on uncompiled graph fails)
    # 6. Compile with memory checkpointer
    from langgraph.checkpoint.memory import MemorySaver
    memory = MemorySaver()
    compiled = graph.compile(checkpointer=memory)
    return compiled


# Module-level compiled graph (import this everywhere)
civicai_graph = build_graph()


def run_graph(query: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Public entry point. Runs the graph and returns the final state.

    FIX: Wraps graph.invoke() in try/except to return structured errors
         instead of letting 500s bubble to FastAPI unhandled.
    """
    initial_state: AgentState = {
        "query":        query,
        "language":     "en",
        "intent":       "",
        "context_docs": [],
        "agent_answer": "",
        "final_answer": "",
        "sources":      [],
        "agent_used":   "",
        "confidence":   0.0,
        "error":        "",
        "metadata":     metadata or {},
    }
    try:
        config = {"configurable": {"thread_id": "default"}}
        result = civicai_graph.invoke(initial_state, config=config)
        return result
    except Exception as exc:
        return {
            **initial_state,
            "final_answer": f"Graph execution error: {exc}",
            "error":        str(exc),
        }
