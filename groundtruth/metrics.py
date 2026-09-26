"""Retrieval metrics against the lawyers' labelled spans.

A chunk is relevant when it holds at least 20 characters of a labelled span for
the query (or the whole span, if it is shorter). Recall is counted per span, not
per chunk: a span counts as found when the top-k chunks together cover at least
half of it. That keeps recall comparable across chunk sizes, where counting
relevant chunks would reward small chunks for splitting one clause into several.
"""
import math

MIN_OVERLAP = 20


def overlap(chunk, span):
    return max(0, min(chunk.end, span[1]) - max(chunk.start, span[0]))


def is_relevant(chunk, doc, spans):
    return chunk.doc == doc and any(overlap(chunk, s) >= min(MIN_OVERLAP, s[1] - s[0]) for s in spans)


def span_recall(ranked, doc, spans, k):
    """Share of labelled spans at least half covered by the top-k chunks together."""
    mine = sorted((c.start, c.end) for c in ranked[:k] if c.doc == doc)
    found = 0
    for s, e in spans:
        covered, cursor = 0, s
        for a, b in mine:
            a, b = max(a, cursor), min(b, e)
            if b > a:
                covered += b - a
                cursor = b
        found += covered * 2 >= e - s
    return found / len(spans)


def mrr(ranked, doc, spans, k=10):
    for rank, c in enumerate(ranked[:k], 1):
        if is_relevant(c, doc, spans):
            return 1 / rank
    return 0.0


def ndcg(ranked, doc, spans, n_relevant, k=10):
    dcg = sum(1 / math.log2(r + 1) for r, c in enumerate(ranked[:k], 1) if is_relevant(c, doc, spans))
    ideal = sum(1 / math.log2(r + 1) for r in range(1, min(k, n_relevant) + 1))
    return dcg / ideal if ideal else 0.0
