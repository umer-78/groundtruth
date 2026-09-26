"""Generation on top of retrieval, scored apart from it.

Retrieval metrics say whether the labelled passage reached the model. These two say
what the model did with it:

- faithfulness: the share of the answer's sentences a judge model finds supported by
  the passages the answer was written from;
- answer relevance: how directly the answer addresses the question, judged 1 to 5
  and scaled to 0..1.

Keeping the two layers apart is the point. A wrong answer written from the right
passage is a generation failure that no retrieval change will fix, and a faithful
answer from the wrong passage is a retrieval failure that no prompt will fix.

Both need a model, reached through any OpenAI-compatible endpoint:
GROUNDTRUTH_LLM_URL (e.g. https://api.openai.com/v1), GROUNDTRUTH_LLM_MODEL, GROUNDTRUTH_LLM_KEY.
"""
import json
import os
import re
import urllib.request

SENTENCE = re.compile(r"(?<=[.!?])\s+")


def chat(messages, timeout=60):
    body = json.dumps({"model": os.environ["GROUNDTRUTH_LLM_MODEL"], "messages": messages, "temperature": 0}).encode()
    req = urllib.request.Request(os.environ["GROUNDTRUTH_LLM_URL"].rstrip("/") + "/chat/completions", body,
                                 {"Content-Type": "application/json",
                                  "Authorization": f"Bearer {os.environ.get('GROUNDTRUTH_LLM_KEY', '')}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def answer(question, passages, llm=chat):
    context = "\n\n".join(f"[{i}] {p}" for i, p in enumerate(passages, 1))
    return llm([{"role": "system", "content": "Answer the question using only the numbered passages. "
                                              "If they do not contain the answer, say that they do not."},
                {"role": "user", "content": f"{context}\n\nQuestion: {question}"}])


def faithfulness(text, passages, llm=chat):
    claims = [s for s in SENTENCE.split(text.strip()) if s]
    if not claims:
        return None
    reply = llm([{"role": "system", "content": "For each numbered claim, decide whether the passages support it. "
                                               "Reply with a JSON list of true or false, one per claim, and nothing else."},
                 {"role": "user", "content": "Passages:\n" + "\n\n".join(passages) + "\n\nClaims:\n"
                                             + "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))}])
    verdicts = json.loads(reply[reply.find("["): reply.rfind("]") + 1])
    return sum(bool(v) for v in verdicts[: len(claims)]) / len(claims)


def relevance(question, text, llm=chat):
    reply = llm([{"role": "system", "content": "Rate from 1 to 5 how directly the answer addresses the question. "
                                               "Reply with the number only."},
                 {"role": "user", "content": f"Question: {question}\nAnswer: {text}"}])
    m = re.search(r"[1-5]", reply)
    return (int(m.group()) - 1) / 4 if m else None
