"""python -m groundtruth eval | gate | review | generate | demo | train-reranker | build-golden"""
import argparse
import csv
import json
import sys
from pathlib import Path

from . import data
from .evaluate import CONFIGS, DEFAULT, RESULTS, overlap_for, run, save, table
from .retrieval import Index
from .search import train_reranker

ROOT = Path(__file__).resolve().parent.parent
SKIP = {"Document Name", "Parties", "Agreement Date", "Effective Date"}  # used to name the contract, or trivially up front


def indexes(sizes):
    docs, _, _ = data.load_contracts()
    return {s: Index(docs, s, overlap_for(s)) for s in sorted(set(sizes))}


def cmd_build_golden(a):
    """Write data/categories.json and data/golden.jsonl from a CUAD checkout."""
    docs, answers, test = data.load_contracts()
    # the description file and the labels spell some names differently ("Cap on Liability", "Cap On Liability")
    label = {k[1].lower(): k[1] for k in answers}
    with open(Path(a.cuad) / "category_descriptions.csv", encoding="utf-8-sig") as f:
        desc = {label.get(n.lower(), n): d for n, d in ((r[0].removeprefix("Category: ").strip(),
                                                         r[1].removeprefix("Description: ").strip()) for r in csv.reader(f))}
    counts = {c: sum(1 for t in test if answers.get((t, c))) for c in desc if c not in SKIP and c in label.values()}
    cats = {c: desc[c] for c in sorted(counts, key=lambda c: (-counts[c], c))[:20]}
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data" / "categories.json").write_text(json.dumps(cats, indent=1))
    text = {d.id: d.text for d in docs}
    with open(ROOT / "data" / "golden.jsonl", "w") as f:
        for q in data.build_queries(docs, answers, test, cats, per_category=5, seed=7):
            f.write(json.dumps({**q.__dict__, "answer": [text[q.doc][s:e] for s, e in q.spans], "reviewed": None}) + "\n")
    print(f"{len(cats)} categories, golden set written")


def cmd_train(a):
    docs, answers, test = data.load_contracts()
    train = [d.id for d in docs if d.id not in test]
    queries = data.build_queries(docs, answers, train, data.categories(), per_category=30, seed=11)
    for size, index in indexes([128, 256, 512]).items():
        model = train_reranker(index, queries)
        (ROOT / "configs").mkdir(exist_ok=True)
        (ROOT / "configs" / f"reranker-{size}.json").write_text(json.dumps(model, indent=1))
        print(size, model["trained_on"], dict(zip(model["features"], model["weights"])))


def cmd_eval(a):
    names = a.configs.split(",") if a.configs else list(CONFIGS)
    golden = data.load_golden()
    built = indexes(CONFIGS[n]["size"] for n in names)
    results = []
    for n in names:
        results.append(run(built[CONFIGS[n]["size"]], golden, n))
        save(results[-1])
        print(f"{n}: {results[-1]['summary']}", flush=True)
    report = table(results)
    if not a.configs:
        (RESULTS / "report.md").write_text(report + "\n")
    print(report)


def cmd_gate(a):
    """Fail when recall@10 of the default configuration drops more than the tolerance below the baseline."""
    baseline = json.loads((RESULTS / "baseline.json").read_text())
    now = run(indexes([CONFIGS[DEFAULT]["size"]])[CONFIGS[DEFAULT]["size"]], data.load_golden(), DEFAULT)
    drop = baseline["summary"]["recall@10"] - now["summary"]["recall@10"]
    print(f"recall@10: baseline {baseline['summary']['recall@10']}, now {now['summary']['recall@10']} ({-drop:+.1f} points)")
    for cat, before in baseline["by_category"].items():
        after = now["by_category"].get(cat)
        if after is not None and after < before:
            print(f"  {cat}: {before} -> {after}")
    if drop > a.tolerance:
        print(f"FAIL: recall@10 dropped {drop:.1f} points, more than the {a.tolerance} allowed")
        sys.exit(1)
    print("ok")


def cmd_review(a):
    """Walk the golden set and record a person's verdict on each query and its labelled passage."""
    path = ROOT / "data" / "golden.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for r in rows:
        if r.get("reviewed"):
            continue
        print(f"\n[{r['qid']}] {r['matter']}: {r['question']}\n  labelled passage: {' / '.join(r['answer'])[:600]}")
        verdict = input("  does the passage answer the question? [y]es / [n]o / [s]kip / [q]uit: ").strip().lower()[:1]
        if verdict == "q":
            break
        if verdict in ("y", "n"):
            r["reviewed"] = {"y": "ok", "n": "rejected"}[verdict]
            path.write_text("".join(json.dumps(x) + "\n" for x in rows))
    done = sum(1 for r in rows if r.get("reviewed"))
    print(f"\n{done}/{len(rows)} reviewed, {sum(r.get('reviewed') == 'rejected' for r in rows)} rejected")


def cmd_generate(a):
    """Answer golden questions from the default configuration's top passages and score the answers."""
    import os
    from . import generate
    from .search import Reranker, search
    if not (os.environ.get("GROUNDTRUTH_LLM_URL") and os.environ.get("GROUNDTRUTH_LLM_MODEL")):
        sys.exit("set GROUNDTRUTH_LLM_URL, GROUNDTRUTH_LLM_MODEL and GROUNDTRUTH_LLM_KEY to an OpenAI-compatible endpoint first")
    cfg, golden = CONFIGS[DEFAULT], data.load_golden()[: a.limit]
    index = indexes([cfg["size"]])[cfg["size"]]
    rows = []
    for q in golden:
        passages = [index.texts[index.chunks.index(c)] for c in
                    search(index, q, cfg["method"], Reranker.load(cfg["size"]), expand=cfg.get("expand", False))[: a.k]]
        text = generate.answer(q.question + f" (in the {q.matter})", passages)
        rows.append({"qid": q.qid, "answer": text, "faithfulness": generate.faithfulness(text, passages),
                     "relevance": generate.relevance(q.text, text)})
        print(rows[-1]["qid"], rows[-1]["faithfulness"], rows[-1]["relevance"], flush=True)
    (RESULTS / "generation.json").write_text(json.dumps(rows, indent=1))


def cmd_demo(a):
    """Rebuild the demo page's data in docs/."""
    from . import demo
    payload = demo.build()
    print(f"wrote docs/data.json: {len(payload['table'])} configurations, {len(payload['questions'])} questions")


def main():
    p = argparse.ArgumentParser(prog="groundtruth")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build-golden"); b.add_argument("--cuad", required=True); b.set_defaults(fn=cmd_build_golden)
    sub.add_parser("train-reranker").set_defaults(fn=cmd_train)
    e = sub.add_parser("eval"); e.add_argument("--configs"); e.set_defaults(fn=cmd_eval)
    g = sub.add_parser("gate"); g.add_argument("--tolerance", type=float, default=1.0); g.set_defaults(fn=cmd_gate)
    sub.add_parser("review").set_defaults(fn=cmd_review)
    n = sub.add_parser("generate"); n.add_argument("--limit", type=int, default=100); n.add_argument("--k", type=int, default=10)
    n.set_defaults(fn=cmd_generate)
    sub.add_parser("demo").set_defaults(fn=cmd_demo)
    a = p.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
