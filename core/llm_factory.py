"""
core/llm_factory.py
────────────────────
FIX 1: HuggingFace HTML response — we use the HF Inference API via
        langchain-community's HuggingFaceEndpoint, NOT the deprecated
        HuggingFaceHub. The endpoint format is:
          https://api-inference.huggingface.co/models/<model_id>
        and we hit it as a text-generation task, not text2text-generation.

FIX 2: flan-t5-base is weak for instruction-following. Default to Groq
        (Llama 3.1 70B free tier) which is production-grade and fast.

FIX 3: No local transformers pipeline — avoids library version mismatch
        entirely. All inference goes through hosted APIs.
"""
from __future__ import annotations
import os
from functools import lru_cache
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


@lru_cache(maxsize=4)
def get_llm(provider: str = "groq", model: str | None = None, temperature: float = 0.1):
    """
    Returns a cached LangChain chat model.

    Providers:
      groq   — Llama 3.1 70B via Groq (free tier, fast, great quality)
      openai — GPT-4o mini (fallback)
      hf     — HuggingFace Inference API (Mistral-7B-Instruct)

    Priority: groq → openai → hf
    """
    if provider == "groq":
        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            temperature=temperature,
            api_key=os.environ["GROQ_API_KEY"],
        )

    if provider == "openai":
        return ChatOpenAI(
            model=model or "gpt-4o-mini",
            temperature=temperature,
            api_key=os.environ["OPENAI_API_KEY"],
        )

    if provider == "hf":
        # FIX: Use HuggingFaceEndpoint, not HuggingFaceHub or pipeline()
        # This returns plain text JSON, never HTML.
        from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
        return HuggingFaceEndpoint(
            endpoint_url=(
                "https://api-inference.huggingface.co/models/"
                + (model or "mistralai/Mistral-7B-Instruct-v0.3")
            ),
            task="text-generation",          # NOT text2text-generation
            max_new_tokens=512,
            temperature=temperature,
            huggingfacehub_api_token=os.environ["HF_API_TOKEN"],
        )

    raise ValueError(f"Unknown provider: {provider!r}. Use 'groq', 'openai', or 'hf'.")


def get_default_llm():
    """Auto-selects best available LLM based on configured env vars."""
    if os.environ.get("GROQ_API_KEY"):
        return get_llm("groq")
    if os.environ.get("OPENAI_API_KEY"):
        return get_llm("openai")
    if os.environ.get("HF_API_TOKEN"):
        return get_llm("hf")
    raise RuntimeError(
        "No LLM API key found. Set GROQ_API_KEY, OPENAI_API_KEY, or HF_API_TOKEN in .env"
    )
