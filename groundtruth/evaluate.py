"""Run retrieval configurations over the golden set and report them side by side."""
import json
import statistics
import time
from pathlib import Path

from .metrics import is_relevant, mrr, ndcg, span_recall
from .search import Reranker, search

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# The three comparisons the gate asks for: chunk size, dense against hybrid, reranker off and on.
CONFIGS = {
    "bm25-256": dict(size=256, method="bm25"),
    "dense-256": dict(size=256, method="dense"),
    "hybrid-256": dict(size=256, method="hybrid"),
    "hybrid-128": dict(size=128, method="hybrid"),
    "hybrid-512": dict(size=512, method="hybrid"),
    "bm25-512": dict(size=512, method="bm25"),
    "dense-512": dict(size=512, method="dense"),
    "hybrid-128-rerank": dict(size=128, method="hybrid", rerank=True),
    "hybrid-256-rerank": dict(size=256, method="hybrid", rerank=True),
    "hybrid-512-rerank": dict(size=512, method="hybrid", rerank=True),
    "hybrid-128-rerank-expand": dict(size=128, method="hybrid", rerank=True, expand=True),
    "hybrid-256-rerank-expand": dict(size=256, method="hybrid", rerank=True, expand=True),
    "hybrid-512-rerank-expand": dict(size=512, method="hybrid", rerank=True, expand=True),
}
DEFAULT = "hybrid-256-rerank-expand"
OVERLAP = 0.125          # of the chunk size
BUDGET_WORDS = 2560      # recall at an equal amount of text handed to the model
METRICS = ["recall@5", "recall@10", "recall@20", "recall@budget", "mrr@10", "ndcg@10"]


def overlap_for(size):
    return int(size * OVERLAP)


def run(index, queries, name):
    cfg = CONFIGS[name]
    reranker = Reranker.load(cfg["size"]) if cfg.get("rerank") else None
    budget_k = max(1, round(BUDGET_WORDS / cfg["size"]))
    rows, started = [], time.perf_counter()
    for q in queries:
        ranked = search(index, q, cfg["method"], reranker, expand=cfg.get("expand", False))
        n_relevant = sum(is_relevant(index.chunks[i], q.doc, q.spans) for i in index.by_doc[q.doc])
        first = next((r for r, c in enumerate(ranked, 1) if is_relevant(c, q.doc, q.spans)), None)
        rows.append({"qid": q.qid, "category": q.category, "first_relevant": first,
                     "recall@5": span_recall(ranked, q.doc, q.spans, 5),
                     "recall@10": span_recall(ranked, q.doc, q.spans, 10),
                     "recall@20": span_recall(ranked, q.doc, q.spans, 20),
                     "recall@budget": span_recall(ranked, q.doc, q.spans, budget_k),
                     "mrr@10": mrr(ranked, q.doc, q.spans), "ndcg@10": ndcg(ranked, q.doc, q.spans, n_relevant),
                     "top3": [f"{c.doc}#{c.index}" for c in ranked[:3]]})
    ms = (time.perf_counter() - started) * 1000 / len(queries)
    pct = lambda xs: round(100 * statistics.fmean(xs), 1)
    categories = sorted({r["category"] for r in rows})
    return {"config": name, **cfg, "chunks": len(index.chunks), "budget_k": budget_k, "ms_per_query": round(ms, 1),
            "summary": {m: pct(r[m] for r in rows) for m in METRICS},
            "by_category": {c: pct(r["recall@10"] for r in rows if r["category"] == c) for c in categories},
            "queries": rows}


def table(results):
    head = "| Configuration | Recall@5 | Recall@10 | Recall@20 | Recall at 2,560 words | MRR@10 | nDCG@10 | ms/query |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
    return head + "\n".join(
        f"| {r['config']} | {r['summary']['recall@5']} | {r['summary']['recall@10']} | {r['summary']['recall@20']} | "
        f"{r['summary']['recall@budget']} | {r['summary']['mrr@10']} | {r['summary']['ndcg@10']} | {r['ms_per_query']} |"
        for r in results)


def save(result):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{result['config']}.json").write_text(json.dumps(result, indent=1))
