import json
from pathlib import Path

import pytest

from groundtruth import generate
from groundtruth.data import Doc, Query, load_golden, party_names
from groundtruth.metrics import is_relevant, mrr, ndcg, span_recall
from groundtruth.retrieval import Chunk, Index, chunk_doc, rrf, top
from groundtruth.search import Reranker, search

ROOT = Path(__file__).resolve().parent.parent


def test_chunks_keep_exact_character_offsets_and_overlap():
    text = " ".join(f"w{i}" for i in range(10))
    chunks = chunk_doc("d", text, size=4, overlap=1)
    assert [text[c.start:c.end] for c in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    assert [c.of for c in chunks] == [3, 3, 3]
    short = "short text"
    assert [short[c.start:c.end] for c in chunk_doc("d", short, 256, 32)] == ["short text"]


def test_a_span_split_across_chunks_counts_when_they_cover_half_of_it_together():
    span = [100, 200]
    first, second = Chunk("d", 0, 140, 0, 2), Chunk("d", 140, 170, 1, 2)
    assert span_recall([first], "d", [span], 10) == 0          # 40 of 100 characters
    assert span_recall([first, second], "d", [span], 10) == 1   # 70 of 100 together
    assert span_recall([Chunk("other", 0, 300, 0, 1)], "d", [span], 10) == 0, "right text, wrong contract"
    assert span_recall([first, second], "d", [span], 1) == 0, "only the top k count"


def test_mrr_and_ndcg_use_the_first_and_all_relevant_chunks():
    spans = [[50, 60]]
    miss, hit = Chunk("d", 0, 40, 0, 3), Chunk("d", 45, 70, 1, 3)
    assert is_relevant(hit, "d", spans) and not is_relevant(miss, "d", spans)
    assert mrr([miss, hit], "d", spans) == 0.5
    assert ndcg([hit, miss], "d", spans, n_relevant=1) == 1.0
    assert 0 < ndcg([miss, hit], "d", spans, n_relevant=1) < 1


def toy_index():
    docs = [Doc("lease", "Lease between Acme Widgets Inc and Beta LLC. " + "filler words here " * 30
                + "This lease is governed by the laws of the State of Texas."),
            Doc("supply", "Supply agreement between Gamma Corp and Delta Ltd. " + "other filler text " * 30
                + "Payment is due within thirty days of invoice."),
            Doc("license", "License between Omega Inc and Zeta Co. " + "license text " * 30
                + "The licence is governed by English law.")]
    return Index(docs, size=40, overlap=4, dims=3)


def test_bm25_finds_the_clause_and_expansion_brings_in_the_named_contract():
    index = toy_index()
    assert "Texas" in index.texts[top(index.bm25_scores("governed laws state texas"), 1)[0]]
    assert index.named_docs({"acme", "widgets", "beta"}, n=1) == ["lease"]
    q = Query("q1", "Governing Law", "Lease between Acme Widgets Inc and Beta LLC",
              "Which State laws govern this lease?", "lease", [[0, 0]])
    prefer_named = Reranker([0, 0, 0, 10, 5, 0], 0)
    ranked = search(index, q, "hybrid", prefer_named, k=1, expand=True)
    assert ranked[0].doc == "lease" and "Texas" in index.texts[index.chunks.index(ranked[0])]
    assert len(rrf(index.bm25_scores(q.text), index.dense_scores(q.text)).nonzero()[0]) > 0


def test_party_names_skip_roles_and_fragments():
    assert party_names(["Distributor", "Google", "Google Inc", "Whitesmoke Inc."]) == ["Whitesmoke Inc.", "Google Inc"]
    assert party_names(['"you" or the "Franchisee"', "Goosehead Insurance Agency, LLC"]) == ["Goosehead Insurance Agency, LLC"]
    assert party_names(["Company", "The seller:", "Licensee"]) == []


def test_the_golden_set_is_100_queries_five_per_category_with_labelled_spans():
    rows = [json.loads(line) for line in (ROOT / "data" / "golden.jsonl").read_text().splitlines()]
    assert len(rows) == 100 and len({r["qid"] for r in rows}) == 100
    per_category = {}
    for r in rows:
        per_category[r["category"]] = per_category.get(r["category"], 0) + 1
        assert r["spans"] and all(0 <= s < e for s, e in r["spans"]) and len(r["answer"]) == len(r["spans"])
    assert len(per_category) == 20 and set(per_category.values()) == {5}
    assert all(q.text.endswith(q.question) for q in load_golden())


def test_generation_scores_are_separate_from_retrieval_and_parse_the_judge():
    judge = lambda messages: "[true, false]" if "Claims" in messages[-1]["content"] else "4"
    assert generate.faithfulness("It is Texas law. It was signed in 1999.", ["governed by the laws of Texas"], llm=judge) == 0.5
    assert generate.relevance("Which law governs?", "Texas law.", llm=judge) == 0.75
    assert generate.faithfulness("", ["x"], llm=judge) is None
    with pytest.raises(KeyError):
        generate.chat([{"role": "user", "content": "hi"}])  # no endpoint configured: refuse, do not guess
