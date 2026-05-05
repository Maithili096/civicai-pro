"""
dashboard.py
─────────────
Streamlit dashboard for CivicAI-Pro.
Shows: live query, agent selector, eval results, vector DB browser.

Install: pip install streamlit
Run:     streamlit run dashboard.py
"""
import streamlit as st
import requests
import json
import time
import pandas as pd

BASE_URL = "http://127.0.0.1:8002"

st.set_page_config(
    page_title="CivicAI-Pro Dashboard",
    page_icon="🏛️",
    layout="wide",
)

st.title("🏛️ CivicAI-Pro — Multilingual Agentic AI")
st.caption("Multi-agent RAG system for civic intelligence")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Config")
    server = st.text_input("Server URL", BASE_URL)
    try:
        r = requests.get(f"{server}/health", timeout=3)
        if r.status_code == 200:
            st.success("✅ Server online")
        else:
            st.error("❌ Server error")
    except:
        st.error("❌ Server offline — run python run.py first")

    st.divider()
    st.header("📋 Agents")
    try:
        agents = requests.get(f"{server}/agents", timeout=3).json()["agents"]
        for a in agents:
            st.markdown(f"**{a['name']}** — {a['description']}")
    except:
        st.warning("Could not load agents")

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📥 Ingest", "🧪 Evals", "🗄️ Vector DB"])

# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Ask a civic question")

    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])

    query = st.chat_input("Type your question (English, Hindi, Marathi...)")
    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.history.append({"role": "user", "content": query})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.time()
                try:
                    r = requests.post(f"{server}/query", json={"query": query}, timeout=30)
                    data = r.json()
                    latency = round((time.time() - start) * 1000)
                    answer = data.get("answer", "No answer")
                    meta = f"🤖 {data.get('agent_used','?')} | 🌐 {data.get('language','?')} | ⚡ {latency}ms | 📎 {', '.join(data.get('sources', [])) or 'no sources'}"
                    st.write(answer)
                    st.caption(meta)
                    if data.get("error"):
                        st.error(data["error"])
                    st.session_state.history.append({"role": "assistant", "content": answer, "meta": meta})
                except Exception as e:
                    st.error(f"Request failed: {e}")

    if st.button("🗑️ Clear chat"):
        st.session_state.history = []
        st.rerun()

# ── Tab 2: Ingest ─────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Add documents to vector store")

    col1, col2 = st.columns(2)
    with col1:
        source_name = st.text_input("Source name", "my_document.txt")
        doc_text = st.text_area("Paste document text here", height=250,
            placeholder="Paste any civic document, policy, scheme details...")

        if st.button("📥 Ingest document", type="primary"):
            if doc_text.strip():
                try:
                    r = requests.post(f"{server}/ingest", json={
                        "texts": [doc_text],
                        "metadatas": [{"source": source_name}]
                    }, timeout=30)
                    result = r.json()
                    st.success(f"✅ Indexed! Total chunks: {result.get('chunks_indexed', '?')}")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")
            else:
                st.warning("Please enter some text first")

    with col2:
        st.info("""
**What to ingest:**
- Government scheme PDFs (copy-paste text)
- Policy documents
- FAQ documents
- News articles about civic topics
- Any multilingual civic content

**The RAG agent will then cite these as sources in answers.**
        """)

        st.subheader("Quick sample data")
        if st.button("📦 Load sample civic docs"):
            samples = [
                {"text": "PM Kisan Samman Nidhi provides Rs 6000 per year to farmers in three installments of Rs 2000 each. Farmers with up to 2 hectares land are eligible.", "source": "pm_kisan.txt"},
                {"text": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana provides health coverage of Rs 5 lakh per family per year. Over 10 crore poor families are covered.", "source": "ayushman_bharat.txt"},
                {"text": "Smart Cities Mission aims to develop 100 cities with modern infrastructure, technology integration, and sustainable development across India.", "source": "smart_cities.txt"},
                {"text": "Swachh Bharat Abhiyan launched in 2014 aims to achieve open defecation free India and improve solid waste management in urban and rural areas.", "source": "swachh_bharat.txt"},
                {"text": "Pradhan Mantri Awas Yojana provides financial assistance for housing to urban and rural poor. The scheme aims to provide housing for all by 2024.", "source": "pmay.txt"},
            ]
            for s in samples:
                requests.post(f"{server}/ingest", json={"texts": [s["text"]], "metadatas": [{"source": s["source"]}]})
            st.success("✅ 5 sample documents ingested!")

# ── Tab 3: Evals ──────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Agent evaluation")
    st.info("Runs 6 test queries and scores your agents on accuracy, latency, and language detection.")

    if st.button("▶️ Run evaluations", type="primary"):
        eval_cases = [
            {"query": "What is PM Kisan scheme?",             "expected": "civic_qa"},
            {"query": "Summarize Ayushman Bharat scheme",     "expected": "summarizer"},
            {"query": "पीएम किसान योजना क्या है?",          "expected": "civic_qa"},
            {"query": "Translate: Good morning to Hindi",     "expected": "nlp_agent"},
            {"query": "What is Swachh Bharat Mission?",       "expected": "civic_qa"},
            {"query": "Smart cities mission details",         "expected": "rag"},
        ]

        results = []
        progress = st.progress(0)
        for i, case in enumerate(eval_cases):
            start = time.time()
            try:
                r = requests.post(f"{server}/query", json={"query": case["query"]}, timeout=30)
                data = r.json()
                latency = round((time.time() - start) * 1000)
                passed = data.get("error","") == "" and len(data.get("answer","")) > 20
                results.append({
                    "Query": case["query"][:45],
                    "Expected": case["expected"],
                    "Got": data.get("agent_used","?"),
                    "Lang": data.get("language","?"),
                    "Confidence": data.get("confidence", 0),
                    "Latency (ms)": latency,
                    "Status": "✅ Pass" if passed else "❌ Fail",
                })
            except Exception as e:
                results.append({
                    "Query": case["query"][:45], "Expected": case["expected"],
                    "Got": "ERROR", "Lang": "?", "Confidence": 0,
                    "Latency (ms)": 0, "Status": f"❌ {str(e)[:30]}",
                })
            progress.progress((i+1)/len(eval_cases))

        df = pd.DataFrame(results)
        passed = sum(1 for r in results if "Pass" in r["Status"])
        total = len(results)
        avg_latency = round(sum(r["Latency (ms)"] for r in results) / total)

        col1, col2, col3 = st.columns(3)
        col1.metric("Pass rate", f"{round(passed/total*100)}%")
        col2.metric("Passed", f"{passed}/{total}")
        col3.metric("Avg latency", f"{avg_latency}ms")

        st.dataframe(df, use_container_width=True)

        st.download_button(
            "💾 Download report",
            data=df.to_csv(index=False),
            file_name="civicai_eval_report.csv",
            mime="text/csv",
        )

# ── Tab 4: Vector DB ──────────────────────────────────────────────────────────
with tab4:
    st.subheader("Vector database browser")

    col1, col2 = st.columns([3,1])
    with col1:
        search_query = st.text_input("Search documents", placeholder="Enter query to find similar docs...")
    with col2:
        k = st.number_input("Top K results", min_value=1, max_value=10, value=4)

    if st.button("🔍 Search vector DB") and search_query:
        try:
            r = requests.post(f"{server}/query", json={"query": search_query}, timeout=30)
            data = r.json()
            st.write("**Retrieved sources:**", data.get("sources", []))
            st.write("**Answer:**", data.get("answer",""))
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.caption("Vector DB is stored at `D:\\civicai_chroma_db` or `civicai_faiss_index` depending on which pipeline is active.")
    st.info("To view all indexed docs, run this in terminal:\n```\nD:\\civicai_pro\\venv\\Scripts\\python.exe -c \"from rag.pipeline import load_index; s=load_index(); print('Docs:', s.index.ntotal if s else 0)\"\n```")
