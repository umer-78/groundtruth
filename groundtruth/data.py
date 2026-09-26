"""CUAD: 510 commercial contracts filed with the SEC, with clauses labelled by lawyers.

Every contract is in the corpus. The golden set, and the queries the reranker
trains on, come from CUAD's labels: each query is CUAD's own question for one
clause category, aimed at one named contract, and the passages that answer it
are the spans the annotators highlighted. No query or label is written by a model.
"""
import hashlib
import json
import os
import random
import re
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

CUAD_COMMIT = "67faa0e6023b04fcaae6cc09497ab00e5d63a2a2"
CUAD_URL = f"https://github.com/TheAtticusProject/cuad/raw/{CUAD_COMMIT}/data.zip"
CUAD_SHA256 = "f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a"
CACHE = Path(os.environ.get("GROUNDTRUTH_CACHE", Path.home() / ".cache" / "groundtruth"))
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Doc:
    id: str
    text: str


@dataclass
class Query:
    qid: str
    category: str
    matter: str        # the contract the question names
    question: str      # CUAD's own wording for the clause category
    doc: str
    spans: list = field(default_factory=list)  # [start, end] character offsets in the doc

    @property
    def text(self):
        return f"{self.matter}: {self.question}"


def fetch_cuad() -> Path:
    """Download CUAD once, pinned to a commit and checked against its hash."""
    CACHE.mkdir(parents=True, exist_ok=True)
    zpath, jpath = CACHE / "cuad-data.zip", CACHE / "CUADv1.json"
    if not jpath.exists():
        if not zpath.exists():
            urllib.request.urlretrieve(CUAD_URL, zpath)
        digest = hashlib.sha256(zpath.read_bytes()).hexdigest()
        if digest != CUAD_SHA256:
            zpath.unlink()
            raise RuntimeError(f"CUAD download has sha256 {digest}, expected {CUAD_SHA256}")
        with zipfile.ZipFile(zpath) as z:
            for name in ("CUADv1.json", "test.json"):
                (CACHE / name).write_bytes(z.read(name))
    return CACHE


def load_contracts():
    """All 510 contracts, their labelled answers, and the names of the 102 test contracts."""
    d = fetch_cuad()
    data = json.loads((d / "CUADv1.json").read_text())["data"]
    test = {c["title"] for c in json.loads((d / "test.json").read_text())["data"]}
    docs, answers = [], {}
    for c in data:
        p = c["paragraphs"][0]
        docs.append(Doc(c["title"], p["context"]))
        for qa in p["qas"]:
            category = qa["id"].split("__")[-1]
            answers[(c["title"], category)] = [(a["answer_start"], a["answer_start"] + len(a["text"])) for a in qa["answers"]]
    return docs, answers, test


# ------------------------------------------------------------------ queries
ENTITY = re.compile(r"\b(inc|incorporated|ltd|limited|llc|l\.l\.c|corp|corporation|co|company|plc|lp|l\.p|gmbh|ag|sa|s\.a|nv|bv|holdings|group|trust)\b\.?", re.I)


SMALL = {"of", "and", "&", "de", "la", "du", "von", "van", "der"}
ROLES = {"company", "corporation", "agent", "distributor", "licensee", "licensor", "supplier", "customer",
         "consultant", "franchisee", "franchisor", "buyer", "seller", "purchaser", "contractor", "client",
         "vendor", "reseller", "manufacturer", "provider", "party", "sponsor", "endorser", "trustee", "bank",
         "investor", "partner", "member", "owner", "operator", "developer", "publisher", "affiliate", "parent"}


def looks_like_name(t):
    words = t.split()
    return 1 <= len(words) <= 8 and not any(q in t for q in "\"“”") and not t.endswith(":") \
        and all(w[0].isupper() or w.lower() in SMALL for w in words if w[0].isalpha()) \
        and any(re.sub(r"\W", "", w).lower() not in ROLES and not ENTITY.fullmatch(w) for w in words)


def party_names(texts):
    """Up to two party names from CUAD's Parties labels, which mix names with roles
    like 'Distributor' and with fragments like '"you" or the "Franchisee"'."""
    def score(t):
        return (2 if ENTITY.search(t) else 0) + (1 if len(t.split()) >= 2 else 0)
    picked = []
    for t in sorted({t.strip() for t in texts if looks_like_name(t.strip())}, key=lambda t: (-score(t), -len(t))):
        if any(t.lower() in p.lower() or p.lower() in t.lower() for p in picked):
            continue
        picked.append(t)
        if len(picked) == 2:
            break
    return picked


def matter_for(doc: Doc, answers) -> str:
    names = [doc.text[s:e] for s, e in answers.get((doc.id, "Document Name"), [])]
    title = names[0].strip() if names else doc.id.split("_")[-1]
    title = title.title() if title.isupper() else title
    parties = party_names([doc.text[s:e] for s, e in answers.get((doc.id, "Parties"), [])])
    if len(parties) == 2:
        return f"{title} between {parties[0]} and {parties[1]}"
    return f"{title} with {parties[0]}" if parties else title


def build_queries(docs, answers, titles, categories, per_category, seed):
    """Queries for (contract, category) pairs that have a labelled answer, drawn evenly across categories."""
    rng = random.Random(seed)
    by_id = {d.id: d for d in docs}
    out = []
    for cat, question in categories.items():
        pool = sorted(t for t in titles if answers.get((t, cat)))
        for t in rng.sample(pool, min(per_category, len(pool))):
            slug = re.sub(r"[^a-z0-9]+", "-", cat.lower()).strip("-")
            out.append(Query(f"{slug}-{len(out):03d}", cat, matter_for(by_id[t], answers), question, t,
                             [list(s) for s in answers[(t, cat)]]))
    return out


def load_golden(path=ROOT / "data" / "golden.jsonl"):
    return [Query(**{k: v for k, v in json.loads(line).items() if k in Query.__dataclass_fields__})
            for line in path.read_text().splitlines() if line.strip()]


def categories(path=ROOT / "data" / "categories.json"):
    return json.loads(path.read_text())
