# CivicAI-Pro v2.0 — Multilingual Agentic AI

## What was fixed (your 7 error categories)

| Your error | Root cause | Fix in this codebase |
|---|---|---|
| FastAPI 500 from `run_graph` | LangGraph exceptions not caught | `run_graph()` wraps `invoke()` in try/except; returns structured error state |
| HuggingFace returning HTML | Used deprecated `HuggingFaceHub` + wrong task `text2text-generation` | `core/llm_factory.py` uses `HuggingFaceEndpoint` with `task="text-generation"` |
| LangGraph nodes returning non-dicts | Nodes returned strings / None | Every node in `agents/nodes.py` returns `dict[str, Any]` explicitly |
| Missing `agents.model` import | Wrong module structure | All imports follow `agents.nodes`, `agents.graph`, `core.llm_factory` |
| transformers `text2text-generation` mismatch | local pipeline() version conflicts | **No local transformers pipeline at all** — all inference through hosted APIs |
| flan-t5-base weak instruction-following | Model too small for structured output | Default is **Groq Llama 3.1 70B** — free, fast, production-grade |
| Unstructured model output | flan-t5 ignores format instructions | Groq/GPT models follow JSON + format instructions reliably |

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env: add your GROQ_API_KEY (free at console.groq.com)

# 3. Run
python run.py
# → http://localhost:8000/docs
```

## API endpoints

### POST /query
```json
{
  "query": "What is PM Kisan Samman Nidhi scheme?",
  "metadata": {}
}
```
Response:
```json
{
  "query": "...",
  "answer": "PM Kisan is...",
  "language": "en",
  "agent_used": "civic_qa",
  "sources": ["doc_0"],
  "confidence": 0.87,
  "error": ""
}
```

### POST /ingest
```json
{
  "texts": ["PM Kisan scheme provides Rs 6000 per year..."],
  "metadatas": [{"source": "pm_kisan_guidelines.pdf"}]
}
```

### GET /agents
Lists all registered agents and their intents.

---

## Architecture

```
FastAPI /query
    └── LangGraph StateGraph (compiled)
            ├── router_node          ← detects language + intent
            ├── [conditional edges]
            │       ├── rag_agent    ← FAISS retrieval + LLM
            │       ├── nlp_agent    ← classify / translate
            │       ├── summarizer   ← bullet-point summaries
            │       └── civic_qa     ← policy + scheme QA
            └── aggregator_node      ← merges to final_answer
```

## Critical LangGraph rules (memorize these)

1. **Every node returns `dict`** — never None, never a string
2. **Conditional edge keys must match node names exactly** — typos cause `KeyError`
3. **Always call `graph.compile()`** before `graph.invoke()`
4. **StateGraph schema uses `TypedDict`** — keys must exist in `AgentState`
5. **Entry point must be set** with `graph.set_entry_point()`

## Adding a new agent

```python
# 1. Add node in agents/nodes.py
def my_new_node(state: AgentState) -> dict[str, Any]:
    ...
    return {"agent_answer": answer, "agent_used": "my_agent", "confidence": 0.9, "error": ""}

# 2. Register in agents/graph.py
graph.add_node("my_agent", my_new_node)

# 3. Add to conditional edges mapping
graph.add_conditional_edges("router", _route_by_intent, {
    ...,
    "my_agent": "my_agent",
})

# 4. Add edge to aggregator
graph.add_edge("my_agent", "aggregator")

# 5. Update router_node prompt to include new intent
```

## Supported languages (via multilingual embeddings)
English, Hindi (हिंदी), Marathi (मराठी), Gujarati, Tamil, Telugu, Bengali, and 50+ more via `paraphrase-multilingual-MiniLM-L12-v2`.
