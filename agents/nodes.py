"""
agents/nodes.py
────────────────
FIX: Every LangGraph node function MUST return a dict.
     Returning None, a string, or raising an unhandled exception
     causes a 500 at graph.invoke() time.

Each node receives the full AgentState dict and returns a partial
update dict — LangGraph merges it with existing state.
"""
from __future__ import annotations
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from core.llm_factory import get_default_llm
from core.state import AgentState
from rag.pipeline import retrieve


# ─── Shared LLM (initialized once) ───────────────────────────────────────────

def _llm():
    """Lazy-loaded so import errors surface at call time, not import time."""
    return get_default_llm()


# ─── Node 1: Language Detection & Intent Router ───────────────────────────────

def router_node(state: AgentState) -> dict[str, Any]:
    """
    Detects language and classifies intent.
    Returns: {language, intent}
    """
    query = state.get("query", "")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a multilingual classifier for a civic AI assistant. "
         "Return ONLY a JSON object with keys 'language' (ISO 639-1 code) "
         "and 'intent' (one of: rag, nlp, summarize, civic_qa). "
         "No markdown, no explanation."),
        ("human", "{query}"),
    ])
    try:
        chain = prompt | _llm()
        result = chain.invoke({"query": query})
        import json, re
        text = result.content if hasattr(result, "content") else str(result)
        # Strip any accidental markdown fences
        text = re.sub(r"```[a-z]*|```", "", text).strip()
        parsed = json.loads(text)
        return {
            "language": parsed.get("language", "en"),
            "intent":   parsed.get("intent",   "rag"),
            "error":    "",
        }
    except Exception as exc:
        return {"language": "en", "intent": "rag", "error": str(exc)}


# ─── Node 2: RAG Agent ────────────────────────────────────────────────────────

def rag_agent_node(state: AgentState) -> dict[str, Any]:
    """
    Retrieves relevant docs and generates a grounded answer.
    Returns: {context_docs, agent_answer, sources, agent_used, confidence}
    """
    query   = state.get("query", "")
    lang    = state.get("language", "en")
    chunks, sources = retrieve(query, k=4)

    context = "\n\n".join(chunks) if chunks else "No relevant documents found."

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a civic assistant. Answer using ONLY the provided context. "
         f"Respond in language code: {lang}. "
         "If context is insufficient, say so clearly."),
        ("human", "Context:\n{context}\n\nQuestion: {query}"),
    ])
    try:
        chain = prompt | _llm()
        result = chain.invoke({"context": context, "query": query})
        answer = result.content if hasattr(result, "content") else str(result)
        return {
            "context_docs": chunks,
            "agent_answer": answer,
            "sources":      sources,
            "agent_used":   "rag_agent",
            "confidence":   0.85 if chunks else 0.3,
            "error":        "",
        }
    except Exception as exc:
        return {
            "context_docs": [],
            "agent_answer": f"RAG agent error: {exc}",
            "sources":      [],
            "agent_used":   "rag_agent",
            "confidence":   0.0,
            "error":        str(exc),
        }


# ─── Node 3: NLP Agent (classify / translate / POS) ──────────────────────────

def nlp_agent_node(state: AgentState) -> dict[str, Any]:
    """
    Handles NLP tasks: classification, translation, sentiment, NER.
    Returns: {agent_answer, agent_used, confidence}
    """
    query = state.get("query", "")
    lang  = state.get("language", "en")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a multilingual NLP specialist. Perform the requested NLP task "
         f"and respond in language: {lang}. Be concise and structured."),
        ("human", "{query}"),
    ])
    try:
        chain = prompt | _llm()
        result = chain.invoke({"query": query})
        answer = result.content if hasattr(result, "content") else str(result)
        return {"agent_answer": answer, "agent_used": "nlp_agent", "confidence": 0.9, "error": ""}
    except Exception as exc:
        return {"agent_answer": f"NLP agent error: {exc}", "agent_used": "nlp_agent", "confidence": 0.0, "error": str(exc)}


# ─── Node 4: Summarizer Agent ─────────────────────────────────────────────────

def summarizer_node(state: AgentState) -> dict[str, Any]:
    """
    Compresses long civic documents into bullet-point summaries.
    Returns: {agent_answer, agent_used, confidence}
    """
    query = state.get("query", "")
    lang  = state.get("language", "en")
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"Summarize the following civic content in {lang}. "
         "Use bullet points. Keep it under 200 words."),
        ("human", "{query}"),
    ])
    try:
        chain = prompt | _llm()
        result = chain.invoke({"query": query})
        answer = result.content if hasattr(result, "content") else str(result)
        return {"agent_answer": answer, "agent_used": "summarizer", "confidence": 0.88, "error": ""}
    except Exception as exc:
        return {"agent_answer": f"Summarizer error: {exc}", "agent_used": "summarizer", "confidence": 0.0, "error": str(exc)}


# ─── Node 5: Civic QA Agent ───────────────────────────────────────────────────

def civic_qa_node(state: AgentState) -> dict[str, Any]:
    """
    Answers policy, legal, scheme, and government-scheme questions.
    Returns: {agent_answer, agent_used, confidence}
    """
    query = state.get("query", "")
    lang  = state.get("language", "en")
    chunks, sources = retrieve(query, k=3)
    context = "\n\n".join(chunks) if chunks else ""

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are a civic and government policy expert. "
         f"Respond in {lang}. Use provided context if available. "
         "Cite government schemes or policies by name when relevant."),
        ("human", "Context:\n{context}\n\nQuestion: {query}"),
    ])
    try:
        chain = prompt | _llm()
        result = chain.invoke({"context": context, "query": query})
        answer = result.content if hasattr(result, "content") else str(result)
        return {
            "agent_answer": answer,
            "sources":      sources,
            "agent_used":   "civic_qa",
            "confidence":   0.87,
            "error":        "",
        }
    except Exception as exc:
        return {"agent_answer": f"CivicQA error: {exc}", "agent_used": "civic_qa", "confidence": 0.0, "error": str(exc)}


# ─── Node 6: Response Aggregator ─────────────────────────────────────────────

def aggregator_node(state: AgentState) -> dict[str, Any]:
    """
    Final node: copies agent_answer → final_answer.
    Add post-processing here (formatting, safety filter, etc.)
    Returns: {final_answer}
    """
    answer = state.get("agent_answer", "I could not generate a response.")
    return {"final_answer": answer}
