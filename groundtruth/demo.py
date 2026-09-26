"""Builds the demo page's data: every configuration's scores, recall by clause type, and for
each golden question the top five passages that plain BM25 and the recommended
configuration return, each marked as the labelled passage, the right contract, or neither.

The contract text is from CUAD (The Atticus Project, CC BY 4.0).

    python -m groundtruth demo     writes docs/data.json
"""
import json

from . import data
from .evaluate import CONFIGS, DEFAULT, RESULTS, ROOT, overlap_for
from .metrics import is_relevant
from .retrieval import Index
from .search import Reranker, search

BASELINE = "bm25-256"
TOP = 5


def excerpt(text, limit=420):
    t = " ".join(text.split())
    return t if len(t) <= limit else t[:limit].rsplit(" ", 1)[0] + " …"


def around(doc_text, chunk, spans, limit=420):
    """The chunk's text, starting just before the labelled passage when the chunk holds one,
    so a hit shows the clause that made it a hit rather than the words before it."""
    inside = [(max(chunk.start, a), min(chunk.end, b)) for a, b in spans if a < chunk.end and b > chunk.start]
    start = max(chunk.start, inside[0][0] - 80) if inside else chunk.start
    return ("… " if start > chunk.start else "") + excerpt(doc_text[start:chunk.end], limit)


def build(out=ROOT / "docs"):
    docs, _, _ = data.load_contracts()
    index = Index(docs, 256, overlap_for(256))
    doc_text = {d.id: d.text for d in docs}
    results = {name: json.loads((RESULTS / f"{name}.json").read_text()) for name in CONFIGS}
    runs = {name: CONFIGS[name] for name in (BASELINE, DEFAULT)}
    rerankers = {name: Reranker.load(cfg["size"]) if cfg.get("rerank") else None for name, cfg in runs.items()}
    per_query = {name: {r["qid"]: r for r in results[name]["queries"]} for name in runs}
    questions = []
    for q in data.load_golden():
        row = {"qid": q.qid, "category": q.category, "matter": q.matter, "question": q.question, "doc": q.doc,
               "answer": "", "runs": {}}
        for name, cfg in runs.items():
            ranked = search(index, q, cfg["method"], rerankers[name], expand=cfg.get("expand", False))[:TOP]
            row["runs"][name] = {
                "recall@10": per_query[name][q.qid]["recall@10"], "first": per_query[name][q.qid]["first_relevant"],
                "passages": [{"doc": c.doc, "same": c.doc == q.doc, "hit": is_relevant(c, q.doc, q.spans),
                              "text": around(doc_text[c.doc], c, q.spans if c.doc == q.doc else [])} for c in ranked]}
        questions.append(row)
    golden = {json.loads(line)["qid"]: json.loads(line) for line in (ROOT / "data" / "golden.jsonl").read_text().splitlines() if line.strip()}
    for row in questions:
        row["answer"] = excerpt(" … ".join(golden[row["qid"]]["answer"]), 600)
    table = [{"config": name, "size": r["size"], "method": r["method"], "rerank": bool(r.get("rerank")), "expand": bool(r.get("expand")),
              "ms": r["ms_per_query"], **r["summary"]} for name, r in results.items()]
    categories = sorted(results[DEFAULT]["by_category"])
    out.mkdir(exist_ok=True)
    payload = {"default": DEFAULT, "baseline": BASELINE, "table": table,
               "categories": [{"name": c, BASELINE: results[BASELINE]["by_category"][c], DEFAULT: results[DEFAULT]["by_category"][c]}
                              for c in categories],
               "questions": questions}
    (out / "data.json").write_text(json.dumps(payload, indent=1))
    return payload
