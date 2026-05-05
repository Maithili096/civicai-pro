<div align="center">

# 🏛️ CivicAI-Pro

### Multilingual Agentic AI System for Civic Intelligence

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-purple)](https://langchain-ai.github.io/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/LLM-Llama%203.3%2070B-orange)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**A production-grade GenAI system with multi-agent orchestration, RAG pipeline, multilingual NLP, and real-time civic Q&A — deployed on cloud with CI/CD.**

[Live Demo](https://civicai-pro-production.up.railway.app) • [API Docs](https://civicai-pro-production.up.railway.app/docs) • [LangSmith Traces](#observability)

</div>

---

## 🎯 What it does

CivicAI-Pro answers questions about government schemes, policies, and civic topics in **English, Hindi, Marathi, and 50+ languages**. It automatically detects language and intent, routes to the right specialist agent, retrieves relevant documents, and generates grounded answers with source citations.

**Example queries:**
- `"What is PM Kisan scheme?"` → civic_qa agent answers with sources
- `"पीएम किसान योजना क्या है?"` → detects Hindi, answers in Hindi
- `"Summarize Ayushman Bharat"` → summarizer agent gives bullet points
- `"Translate: Good morning to Marathi"` → nlp_agent translates

---

## 🏗️ Architecture

```
User Query
    │
    ▼
FastAPI REST API (/query, /ingest, /agents)
    │
    ▼
LangGraph StateGraph (Multi-Agent Orchestrator)
    │
    ├── router_node ──────── Language detection + Intent classification
    │        │
    │   [conditional routing]
    │        │
    ├── rag_agent ─────────── FAISS retrieval + Groq LLM
    ├── nlp_agent ─────────── Classification / Translation / NER  
    ├── summarizer ─────────── Bullet-point document summaries
    └── civic_qa ──────────── Policy + Government scheme QA
             │
             ▼
        aggregator_node ───── Merges to final structured response
             │
             ▼
    JSON Response
    { answer, language, agent_used, sources, confidence }
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **LLM** | Groq Llama 3.3 70B | Fast, high-quality inference |
| **Agent Framework** | LangGraph 0.2 | Multi-agent orchestration with state |
| **LLM Orchestration** | LangChain | Chains, prompts, memory |
| **RAG** | FAISS + sentence-transformers | Semantic document retrieval |
| **Embeddings** | paraphrase-multilingual-MiniLM-L12-v2 | 50+ language support |
| **Vector DB** | FAISS / ChromaDB | Persistent document storage |
| **API** | FastAPI + Uvicorn + Pydantic | REST endpoints |
| **Dashboard** | Streamlit | Interactive UI |
| **Memory** | LangGraph MemorySaver | Conversation history |
| **Observability** | LangSmith | Agent tracing + debugging |
| **Container** | Docker + docker-compose | Consistent deployment |
| **CI/CD** | GitHub Actions | Automated test + deploy |
| **Cloud** | Railway | Live deployment |

---

## 🚀 Quick Start

### Local setup

```bash
# 1. Clone
git clone https://github.com/Maithili096/civicai-pro.git
cd civicai-pro

# 2. Install
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Add your GROQ_API_KEY from console.groq.com (free)

# 4. Run
python run.py
# → http://localhost:8002/docs
```

### Docker

```bash
docker-compose up
# API  → http://localhost:8002
# Dashboard → http://localhost:8501
```

---

## 📡 API Reference

### POST /query
```json
Request:
{
  "query": "What is PM Kisan scheme?",
  "metadata": {}
}

Response:
{
  "query": "What is PM Kisan scheme?",
  "answer": "PM Kisan Samman Nidhi provides Rs 6000 per year...",
  "language": "en",
  "agent_used": "civic_qa",
  "sources": ["pm_kisan.txt"],
  "confidence": 0.87,
  "error": ""
}
```

### POST /ingest
```json
{
  "texts": ["Your document text here..."],
  "metadatas": [{"source": "document_name.pdf"}]
}
```

### GET /agents
Returns list of all registered agents with descriptions.

---

## 🧪 Evaluation

Run the built-in evaluation suite:

```bash
python core/evals.py
```

Tests 6 queries across English and Hindi, scoring on:
- Answer quality (length + coherence)
- Agent routing accuracy
- Language detection
- Response latency

---

## 🌍 Supported Languages

English, Hindi (हिंदी), Marathi (मराठी), Gujarati, Tamil, Telugu, Bengali, Punjabi, Urdu, and 50+ more via `paraphrase-multilingual-MiniLM-L12-v2`.

---

## 🔍 Observability

LangSmith tracing enabled — every agent step is logged:

```
router_node        245ms  ✓  intent: civic_qa, language: en
civic_qa_node     4200ms  ✓  retrieved 3 docs
  retrieve()       180ms  ✓  FAISS similarity search
  llm.invoke()    3800ms  ✓  Groq Llama 3.3 70B
aggregator_node     12ms  ✓  final answer assembled
Total:            4457ms
```

---

## 📁 Project Structure

```
civicai-pro/
├── agents/
│   ├── graph.py          # LangGraph StateGraph + routing
│   └── nodes.py          # 4 specialist agent nodes
├── api/
│   └── main.py           # FastAPI endpoints
├── core/
│   ├── state.py          # Shared AgentState TypedDict
│   ├── llm_factory.py    # LLM provider abstraction
│   └── evals.py          # Evaluation framework
├── rag/
│   ├── pipeline.py       # FAISS vector store + retrieval
│   └── vector_db.py      # ChromaDB persistent store
├── dashboard.py          # Streamlit UI
├── run.py                # Server entry point
├── Dockerfile            # Container definition
├── docker-compose.yml    # Multi-service orchestration
└── .github/workflows/    # CI/CD pipelines
```

---

## 🤝 Adding a New Agent

```python
# 1. Add node in agents/nodes.py
def my_agent_node(state: AgentState) -> dict:
    ...
    return {"agent_answer": answer, "agent_used": "my_agent", "confidence": 0.9, "error": ""}

# 2. Register in agents/graph.py
graph.add_node("my_agent", my_agent_node)
graph.add_conditional_edges("router", _route_by_intent, {"my_intent": "my_agent"})
graph.add_edge("my_agent", "aggregator")
```

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built with ❤️ using LangGraph + Groq + FastAPI
</div>
