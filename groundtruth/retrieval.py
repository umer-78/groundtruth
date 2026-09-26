"""Chunking and three retrievers over the same chunks: BM25, dense, and both fused.

The dense retriever is latent semantic analysis: TF-IDF reduced to 256 dimensions
with SVD. It is the dense model measured here because it needs nothing downloaded;
any function that maps texts to unit vectors can take its place in Index.dense_scores.
"""
import re
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import normalize

WORD = re.compile(r"\S+")
TOKEN = re.compile(r"[a-z0-9]+")
STOP = frozenset(ENGLISH_STOP_WORDS)


def tokens(text):
    return [t for t in TOKEN.findall(text.lower()) if t not in STOP]


@dataclass(frozen=True)
class Chunk:
    doc: str
    start: int   # character offsets into the document, so chunks of any size
    end: int     # can be checked against the labelled spans
    index: int
    of: int


def chunk_doc(doc_id, text, size, overlap):
    """Windows of `size` words, each starting `size - overlap` words after the last."""
    words = [m.span() for m in WORD.finditer(text)]
    if not words:
        return []
    starts = range(0, max(1, len(words) - overlap), max(1, size - overlap))
    return [Chunk(doc_id, words[i][0], words[min(i + size, len(words)) - 1][1], n, len(starts))
            for n, i in enumerate(starts)]


class Index:
    def __init__(self, docs, size, overlap, dims=256, seed=0):
        self.chunks = [c for d in docs for c in chunk_doc(d.id, d.text, size, overlap)]
        text = {d.id: d.text for d in docs}
        self.texts = [text[c.doc][c.start:c.end] for c in self.chunks]
        # every word of each contract, to recognise the contract a question names:
        # parties are often named only on the signature pages, not in the opening
        self.terms = {d.id: set(tokens(d.text)) for d in docs}
        df = {}
        for words in self.terms.values():
            for t in words:
                df[t] = df.get(t, 0) + 1
        self.term_idf = {t: np.log(len(docs) / n) for t, n in df.items()}
        self.by_doc = {}
        for i, c in enumerate(self.chunks):
            self.by_doc.setdefault(c.doc, []).append(i)
        self._bm25(k1=1.2, b=0.75)
        self._lsa(dims, seed)

    def _bm25(self, k1, b):
        self.vocab = CountVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None)
        tf = self.vocab.fit_transform(self.texts).tocoo()
        dl = np.bincount(tf.row, weights=tf.data, minlength=len(self.texts))
        df = np.bincount(tf.col, minlength=tf.shape[1])
        idf = np.log(1 + (len(self.texts) - df + 0.5) / (df + 0.5))
        w = idf[tf.col] * tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / dl.mean()))
        self.bm25 = sparse.csc_matrix((w, (tf.row, tf.col)), shape=tf.shape)

    def _lsa(self, dims, seed):
        self.tfidf = TfidfVectorizer(tokenizer=tokens, lowercase=False, token_pattern=None,
                                     sublinear_tf=True, min_df=2, max_df=0.5)
        self.svd = TruncatedSVD(dims, random_state=seed)
        self.vectors = normalize(self.svd.fit_transform(self.tfidf.fit_transform(self.texts))).astype(np.float32)

    def matter_match(self, matter, doc):
        """How much of a contract's name appears in a document, each word weighted by its rarity, 0 to 1."""
        total = sum(self.term_idf.get(t, 0) for t in matter)
        return sum(self.term_idf.get(t, 0) for t in matter & self.terms[doc]) / total if total else 0.0

    def named_docs(self, matter, n=2):
        return sorted(self.terms, key=lambda d: -self.matter_match(matter, d))[:n]

    def bm25_scores(self, text):
        v = self.vocab.vocabulary_
        ids = [v[t] for t in tokens(text) if t in v]
        return np.asarray(self.bm25[:, ids].sum(axis=1)).ravel() if ids else np.zeros(len(self.chunks))

    def dense_scores(self, text):
        q = normalize(self.svd.transform(self.tfidf.transform([text])))
        return self.vectors @ q.ravel().astype(np.float32)


def top(scores, n):
    n = min(n, len(scores))
    idx = np.argpartition(-scores, n - 1)[:n]
    return idx[np.argsort(-scores[idx], kind="stable")]


def rrf(*scores, k=60, depth=200):
    """Reciprocal rank fusion: each list adds 1/(k + rank) for the chunks it ranks."""
    fused = np.zeros(len(scores[0]))
    for s in scores:
        fused[top(s, depth)] += 1 / (k + np.arange(1, min(depth, len(s)) + 1))
    return fused
