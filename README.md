# Groundtruth

[![CI](https://github.com/umer-78/groundtruth/actions/workflows/ci.yml/badge.svg)](https://github.com/umer-78/groundtruth/actions/workflows/ci.yml)

**Live demo:** https://umer-78.github.io/groundtruth/ (every configuration's scores, and the passages
plain BM25 and the recommended configuration return for each of the 100 questions)

**A retrieval evaluation harness for a legal research assistant. 100 questions over 510 real
contracts, each answered by passages that lawyers labelled, so a retrieval change is measured
instead of argued. The configuration it recommends puts the labelled passage in the top 10 for
70.0% of questions, against 36.8% for plain BM25, and CI fails any change that costs more than a
point of that.**

The setting: a law firm's research assistant searches thousands of contracts. Someone changes the
chunk size or adds a reranker every few weeks, and nobody can say whether last month's version was
better. A missed clause is a malpractice conversation, so the answer has to be a number.

## Result

Every configuration answers the same 100 questions over all 510 contracts in
[CUAD](https://www.atticusprojectai.org/cuad), 4.0 million words filed with the SEC.

| Chunks | Retrieval | Reranker | Contract expansion | Recall@10 | Recall@20 | Recall at 2,560 words | MRR@10 | nDCG@10 | ms/query |
|---:|---|:---:|:---:|---:|---:|---:|---:|---:|---:|
| 256 words | BM25 | | | 36.8 | 46.8 | 36.8 | 17.0 | 19.9 | 18 |
| 256 words | dense | | | 25.2 | 31.7 | 25.2 | 18.3 | 17.8 | 18 |
| 256 words | hybrid | | | 32.9 | 41.5 | 32.9 | 19.2 | 20.2 | 16 |
| 512 words | BM25 | | | 50.7 | 56.5 | 35.2 | 22.8 | 27.2 | 17 |
| 512 words | dense | | | 47.3 | 57.2 | 37.8 | 29.2 | 29.8 | 16 |
| 512 words | hybrid | | | 52.0 | 62.6 | 39.8 | 28.8 | 30.6 | 16 |
| 128 words | hybrid | | | 14.5 | 19.0 | 19.0 | 10.6 | 9.9 | 18 |
| 128 words | hybrid | ✓ | | 34.5 | 41.4 | 41.4 | 31.0 | 27.0 | 27 |
| 256 words | hybrid | ✓ | | 62.0 | 71.6 | 62.0 | 45.5 | 43.0 | 31 |
| 512 words | hybrid | ✓ | | 73.2 | 80.6 | 66.9 | 51.2 | 52.4 | 45 |
| 128 words | hybrid | ✓ | ✓ | 49.8 | 59.8 | 59.8 | 38.7 | 36.8 | 38 |
| **256 words** | **hybrid** | **✓** | **✓** | **70.0** | **85.7** | **70.0** | **47.9** | **48.0** | **41** |
| 512 words | hybrid | ✓ | ✓ | 78.4 | 90.5 | 70.8 | 53.2 | 55.6 | 54 |

"Recall at 2,560 words" compares chunk sizes at the same amount of text handed to the model: the
top 20 chunks of 128 words, the top 10 of 256, the top 5 of 512. The numbers for every
configuration and every question are in [`results/`](results).

### Recommendation

Use **256-word chunks, BM25 and dense retrieval fused, a reranker, and the named contract's own
passages added to the reranker's candidates** (the bold row). It finds 70.0% of labelled passages in
the top 10, which hands the model 2,560 words.

The tradeoff is context against recall. At the same 2,560 words, 256- and 512-word chunks tie
(70.0 and 70.8, less than one question apart), and 256 is a quarter faster and cites narrower
passages. 512-word chunks reach 78.4% in the top 10, but that is 5,120 words per question: twice the
tokens at generation. Take that only if the generation model's context and budget allow it. Do not
go below 256: 128-word chunks split clauses apart and fall to 59.8% at the same budget.

### What the numbers say

1. **The question names a contract, so use it.** Retrieval that ignores this returns the right
   kind of clause from the wrong contract: every governing-law clause in the corpus looks alike.
   The reranker's heaviest feature is how well a chunk's contract matches the one the question
   names, and adding that contract's own passages to the candidates raised recall@10 from 62.0 to
   70.0 at 256 words.
2. **A weak dense model can make fusion worse.** The dense retriever here is LSA, because nothing
   could be downloaded where this was built. At 256 words it found 25.2% alone, and fusing it with
   BM25 lowered BM25 from 36.8 to 32.9. At 512 words it is stronger (47.3) and fusion helps (52.0,
   and 62.6 against 56.5 at recall@20). Test a dense model on its own before fusing it.
3. **The reranker is capped by its candidates.** The right passage was in the fused top 50 for 66
   questions and in the top 200 for 79, so reranking 200 candidates instead of 50 raised recall@10
   from 48.7 to 62.0. With contract expansion on, 50 candidates do as well as 200 (70.0 either way)
   and are 23% faster, 32 against 42 ms per question.

### Recall@10 by clause category (recommended configuration)

| Category | Recall@10 | | Category | Recall@10 |
|---|---:|---|---|---:|
| Anti-Assignment | 100 | | Non-Transferable License | 70 |
| Expiration Date | 100 | | Volume Restriction | 70 |
| Insurance | 100 | | IP Ownership Assignment | 61 |
| Renewal Term | 100 | | Governing Law | 60 |
| Change Of Control | 90 | | Non-Compete | 56 |
| Cap On Liability | 83 | | Minimum Commitment | 47 |
| Audit Rights | 80 | | Covenant Not To Sue | 40 |
| Termination For Convenience | 80 | | Rofr/Rofo/Rofn | 40 |
| License Grant | 77 | | Revenue/Profit Sharing | 37 |
| Post-Termination Services | 76 | | Exclusivity | 33 |

## Error analysis: five questions that still fail

23 of the 100 questions have no labelled passage in the recommended top 10. In 18 of them the
right contract is found (its passages fill the top three) but the clause is not: 16 times it ranks
between 11th and 117th, and twice the top 10 hold less than half of it because a chunk boundary
cuts through it. In the other 5 the contract itself is not found, and all 5 are the same contract.
Five that show why:

1. **`governing-law-001`**, Co-Branding Agreement with About.com, Inc. Once "about" is dropped as a
   stop word the name is "com" and "inc", which half the corpus shares, so the contract is never
   identified and a different co-branding agreement comes first. All five "contract not found"
   failures are this contract. Its clause does not even name a state: "the state where a suit is
   properly filed".
2. **`rofr-rofo-rofn-091`**, Goosehead franchise agreement, 47,733 words. The right of first
   refusal is written as "prior right and option ... before the effective date of such proposed
   assignment". It ranks 117th among the contract's own passages.
3. **`ip-ownership-assignment-076`**, Babcock & Wilcox intellectual property agreement. The labelled
   clause begins "SpinCo and RemainCo agree and acknowledge that, although RemainCo was responsible
   for ..." and shares no words with CUAD's question about assigning intellectual property. It
   ranks 74th.
4. **`revenue-profit-sharing-033`**, Qualigen and Sekisui distribution agreement. Revenue sharing is
   written as a pricing formula, "The price that Sekisui shall pay for the Reagent Kits Products
   shall be based upon a formula ...", with no word from the question. It ranks 15th.
5. **`exclusivity-038`**, Valeant and Dova co-promotion agreement. A relevant chunk is 2nd, but the
   524-character clause starts at the end of it: that chunk holds its first 85 characters and the
   chunk with the rest is not in the top 10, so under the half-coverage rule the clause counts as
   missed. A chunk boundary, not a ranking failure.

Most of the remaining failures are the question and the clause using different words. The next
change to measure is a model that matches meaning rather than words, a legal embedding model or a
cross-encoder reranker. The harness takes any embedding function, and these 18 questions are what
it has to fix. Chunking on clause boundaries instead of word counts is the fix to test for the fifth.

## How it works

**The golden set** ([`data/golden.jsonl`](data/golden.jsonl)) is 100 questions, five in each of the
20 clause categories with the most labels, drawn at random from CUAD's 102 test contracts. Each
question is CUAD's own wording for its category aimed at one named contract, for example
*"Sponsorship Agreement between Platinum Partners Value Arbitrage Fund L.P. and Snowy August Fund I
LP: Which state/country's law governs the interpretation of the contract?"*. The contract name and
parties come from CUAD's labels, and the answers are the passages its annotators highlighted.
CUAD's labels were made by law students and checked by experienced lawyers. No question or answer
was written by a model. A person at the firm still has to review them before the numbers carry
weight there: `python -m groundtruth review` records a verdict for each pair, and 0 of 100 have been
reviewed so far.

**Metrics.** A chunk is relevant if it holds at least 20 characters of a labelled passage for that
question. Recall@k counts labelled passages, not chunks: a passage counts as found when the top k
chunks together cover at least half of it, which keeps recall comparable across chunk sizes. MRR@10
and nDCG@10 use the relevant chunks. Everything is also reported per category.

**Retrieval.** BM25 (k1 1.2, b 0.75). Dense is LSA: TF-IDF reduced to 256 dimensions. Hybrid fuses
the two with reciprocal rank fusion (k 60). The reranker is a logistic regression over six features
of each of the top 200 candidates: the BM25, dense and fused scores, how well the chunk's contract
matches the one named, how much of the question the chunk contains, and where it sits in its
contract. It is trained on 600 questions built the same way from the 408 training contracts and
never sees the golden set. Contract expansion adds every passage of the two contracts whose text
best matches the named contract, weighting rare words more, since parties are often named only on
the signature pages.

**Generation** ([`groundtruth/generate.py`](groundtruth/generate.py)) is scored apart from
retrieval: faithfulness, the share of an answer's sentences a judge model finds supported by the
passages, and answer relevance. A wrong answer from the right passage is a generation failure that
no retrieval change will fix, so the two are never mixed into one number. It needs a model and was
not run for these results; `python -m groundtruth generate` runs it against any OpenAI-compatible
endpoint.

## The CI gate

`python -m groundtruth gate` runs the recommended configuration and fails when recall@10 falls more
than one point below [`results/baseline.json`](results/baseline.json), listing the categories that
dropped. CI runs it on every push and pull request, after the unit tests.

A change that sounds reasonable, 128-word chunks for more precise citations, fails it:

```
recall@10: baseline 70.0, now 49.8 (-20.2 points)
FAIL: recall@10 dropped 20.2 points, more than the 1.0 allowed
```

That run is the [`demo/smaller-chunks`](https://github.com/umer-78/groundtruth/tree/demo/smaller-chunks)
branch, whose CI passes only when the gate refuses the change this way.

## Run it

```bash
pip install -e '.[dev]'
python -m groundtruth eval        # every configuration; the first run downloads CUAD (18 MB)
python -m groundtruth gate        # the CI check
python -m groundtruth review      # record a verdict on each golden question
python -m groundtruth demo        # rebuild the demo page's data in docs/
pytest -q
```

`python -m groundtruth train-reranker` retrains the rerankers from the training contracts, and
`python -m groundtruth build-golden --cuad <checkout>` rebuilds the golden set from a CUAD checkout.

## Data and licence

Code: MIT. CUAD is by The Atticus Project and licensed CC BY 4.0; the golden set quotes its
labelled passages. The dataset is downloaded from a pinned commit and checked against its SHA-256.
