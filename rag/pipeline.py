from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_INDEX_PATH = "civicai_faiss_index"

# Load ONCE at module import — not on every query
print("[RAG] Loading embedding model...")
_EMBEDDINGS = HuggingFaceEmbeddings(
    model_name=_EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
print("[RAG] Embedding model ready.")

def build_index(texts: list[str], metadatas: Optional[list[dict]] = None) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "।", ".", " "],
    )
    docs: list[Document] = []
    for i, text in enumerate(texts):
        chunks = splitter.create_documents(
            [text],
            metadatas=[metadatas[i]] if metadatas else [{"source": f"doc_{i}"}],
        )
        docs.extend(chunks)
    store = FAISS.from_documents(docs, _EMBEDDINGS)
    store.save_local(_INDEX_PATH)
    print(f"[RAG] Built index: {len(docs)} chunks")
    return store

def load_index() -> Optional[FAISS]:
    if Path(_INDEX_PATH).exists():
        return FAISS.load_local(
            _INDEX_PATH,
            _EMBEDDINGS,
            allow_dangerous_deserialization=True,
        )
    return None

def retrieve(query: str, k: int = 4) -> tuple[list[str], list[str]]:
    store = load_index()
    if store is None:
        return [], []
    results = store.similarity_search_with_score(query, k=k)
    chunks = [doc.page_content for doc, _ in results]
    sources = [doc.metadata.get("source", "unknown") for doc, _ in results]
    return chunks, sources