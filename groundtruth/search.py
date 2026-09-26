"""One retrieval configuration end to end: a retriever, then an optional reranker.

The reranker is a logistic regression over six features of each of the top 200
candidates. It is trained on queries built from the 408 training contracts and
never sees the golden set, which comes from the 102 test contracts.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from .metrics import is_relevant
from .retrieval import rrf, tokens, top

FEATURES = ["bm25", "dense", "fused", "matter", "question", "position"]
DEPTH = 200
ROOT = Path(__file__).resolve().parent.parent


def features(index, q, order, bm, dn, fused):
    """bm25 (relative to the best candidate), dense cosine, fused score, how much of the
    named contract the chunk's document matches, how much of the question the chunk
    contains, and where in its document the chunk sits."""
    matter, question = set(tokens(q.matter)), set(tokens(q.question))
    best = bm[order].max() or 1.0
    rows = []
    for i in order:
        c = index.chunks[i]
        rows.append([bm[i] / best, dn[i], fused[i] * 60,
                     index.matter_match(matter, c.doc),
                     len(question & set(tokens(index.texts[i]))) / max(1, len(question)),
                     c.index / c.of])
    return np.array(rows)


class Reranker:
    def __init__(self, weights, bias, **info):
        self.w, self.b, self.info = np.array(weights), bias, info

    @classmethod
    def load(cls, size):
        return cls(**json.loads((ROOT / "configs" / f"reranker-{size}.json").read_text()))

    def order(self, x, order):
        return order[np.argsort(-(x @ self.w + self.b), kind="stable")]


def search(index, q, method="hybrid", reranker=None, k=DEPTH, expand=False):
    """expand: also hand the reranker every chunk of the two contracts whose names best
    match the one the question names, so a clause that shares no words with the
    contract's name can still be found."""
    bm, dn = index.bm25_scores(q.text), index.dense_scores(q.text)
    fused = rrf(bm, dn)
    order = top({"bm25": bm, "dense": dn, "hybrid": fused}[method], k)
    if expand:
        extra = [i for d in index.named_docs(set(tokens(q.matter))) for i in index.by_doc[d]]
        order = np.concatenate([order, np.setdiff1d(extra, order)])
    if reranker is not None:
        order = reranker.order(features(index, q, order, bm, dn, fused), order)
    return [index.chunks[i] for i in order]


def train_reranker(index, queries):
    xs, ys = [], []
    for q in queries:
        bm, dn = index.bm25_scores(q.text), index.dense_scores(q.text)
        fused = rrf(bm, dn)
        order = top(fused, DEPTH)
        xs.append(features(index, q, order, bm, dn, fused))
        ys += [is_relevant(index.chunks[i], q.doc, q.spans) for i in order]
    x, y = np.vstack(xs), np.array(ys)
    model = LogisticRegression(class_weight="balanced", max_iter=2000).fit(x, y)
    return {"weights": [round(float(w), 6) for w in model.coef_[0]], "bias": round(float(model.intercept_[0]), 6),
            "features": FEATURES, "trained_on": {"queries": len(queries), "candidates": int(len(y)), "relevant": int(y.sum())}}
