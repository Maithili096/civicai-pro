"""
core/evals.py
──────────────
Agent evaluation framework.
Run: python core/evals.py
"""
from __future__ import annotations
import time, json, statistics
from dataclasses import dataclass, field, asdict

@dataclass
class EvalResult:
    query:          str
    expected_agent: str
    actual_agent:   str
    answer:         str
    sources:        list[str]
    confidence:     float
    latency_ms:     float
    language:       str
    passed:         bool
    error:          str = ""

@dataclass
class EvalReport:
    total:       int   = 0
    passed:      int   = 0
    failed:      int   = 0
    avg_latency: float = 0.0
    pass_rate:   float = 0.0
    results:     list  = field(default_factory=list)

EVAL_CASES = [
    {"query": "What is PM Kisan scheme?",               "expected_agent": "civic_qa"},
    {"query": "Summarize the Ayushman Bharat scheme",   "expected_agent": "summarizer"},
    {"query": "पीएम किसान योजना क्या है?",             "expected_agent": "civic_qa"},
    {"query": "Translate: Good morning to Hindi",       "expected_agent": "nlp_agent"},
    {"query": "What is Swachh Bharat Mission?",         "expected_agent": "civic_qa"},
    {"query": "Smart cities development in India",      "expected_agent": "rag"},
]

def run_evals(base_url: str = "http://127.0.0.1:8002") -> EvalReport:
    import requests
    report = EvalReport()
    latencies = []
    for case in EVAL_CASES:
        start = time.time()
        try:
            r = requests.post(f"{base_url}/query", json={"query": case["query"]}, timeout=30)
            latency = (time.time() - start) * 1000
            data = r.json()
            passed = data.get("error","") == "" and len(data.get("answer","")) > 20
            result = EvalResult(
                query=case["query"], expected_agent=case["expected_agent"],
                actual_agent=data.get("agent_used",""), answer=data.get("answer","")[:120],
                sources=data.get("sources",[]), confidence=data.get("confidence",0),
                latency_ms=round(latency,1), language=data.get("language",""),
                passed=passed, error=data.get("error",""),
            )
        except Exception as exc:
            latency = (time.time() - start) * 1000
            result = EvalResult(query=case["query"], expected_agent=case["expected_agent"],
                actual_agent="", answer="", sources=[], confidence=0,
                latency_ms=round(latency,1), language="", passed=False, error=str(exc))
        report.results.append(result)
        report.total += 1
        if result.passed: report.passed += 1
        else: report.failed += 1
        latencies.append(result.latency_ms)

    report.avg_latency = round(statistics.mean(latencies), 1)
    report.pass_rate   = round((report.passed / report.total) * 100, 1)
    return report

def print_report(report: EvalReport):
    print("\n" + "="*60)
    print("  CIVICAI-PRO EVAL REPORT")
    print("="*60)
    print(f"  Total:       {report.total}")
    print(f"  Passed:      {report.passed}  PASS")
    print(f"  Failed:      {report.failed}  FAIL")
    print(f"  Pass rate:   {report.pass_rate}%")
    print(f"  Avg latency: {report.avg_latency}ms")
    print("="*60)
    for r in report.results:
        status = "PASS" if r.passed else "FAIL"
        print(f"\n  [{status}] {r.query[:50]}")
        print(f"    Agent: {r.actual_agent} | Confidence: {r.confidence} | {r.latency_ms}ms")
        if r.error: print(f"    ERROR: {r.error}")
        else: print(f"    Answer: {r.answer[:80]}...")
    with open("eval_report.json","w") as f:
        json.dump(asdict(report), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to eval_report.json\n")

if __name__ == "__main__":
    print("Running CivicAI-Pro evaluations...")
    report = run_evals()
    print_report(report)
