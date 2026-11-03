import json, re, pathlib
import numpy as np, pandas as pd
from rank_bm25 import BM25Okapi

P = pathlib.Path("data/processed"); RES = pathlib.Path("results")
LANGS = ["en", "hi", "ml"]
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
qrels = json.load(open(P / "qrels.json"))

def tok_word(t):
    return [w for w in (x.strip(".,;:!?\"'()[]{}।“”‘’-–—") for x in t.lower().split()) if w]

def tok_ngram(t, n=4):
    out = []
    for w in tok_word(t):
        w = f"#{w}#"
        out += [w[i:i + n] for i in range(max(1, len(w) - n + 1))]
    return out

def zscore(s):
    return (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)

def metrics(sims, gold):
    rank = (sims > sims[np.arange(len(gold)), gold][:, None]).sum(1) + 1
    return {"Recall@1": (rank <= 1).mean(), "Recall@5": (rank <= 5).mean(),
            "Recall@10": (rank <= 10).mean(), "MRR@10": np.where(rank <= 10, 1 / rank, 0).mean()}

rows = []
for dense_name in ["bge-m3", "e5-small"]:
    for lang in LANGS:
        corpus, queries = read(P / f"corpus_{lang}.jsonl"), read(P / f"queries_{lang}.jsonl")
        idx = {c["id"]: i for i, c in enumerate(corpus)}
        gold = np.array([idx[qrels[q["id"]]] for q in queries])
        z = np.load(RES / "emb" / f"{dense_name}_{lang}.npz")
        dense = z["Q"] @ z["D"].T
        for tname, tok in (("word", tok_word), ("ngram4", tok_ngram)):
            bm = BM25Okapi([tok(c["text"]) for c in corpus])
            sparse = np.array([bm.get_scores(tok(q["text"])) for q in queries])
            variants = {"bm25": sparse, "dense": dense}
            for w in (0.3, 0.5, 0.7):  # weight on dense
                variants[f"hybrid_w{w}"] = w * zscore(dense) + (1 - w) * zscore(sparse)
            for vname, s in variants.items():
                if vname == "dense" and tname == "ngram4":
                    continue  # identical to word run
                m = metrics(s, gold)
                m.update(dense_model=dense_name, lang=lang, tokenizer=tname, method=vname)
                rows.append(m)

df = pd.DataFrame(rows)
df.to_csv(RES / "hybrid.csv", index=False)
pd.set_option("display.width", 250)
view = df[(df.method.isin(["bm25", "dense", "hybrid_w0.5"]))].round(3)
print(view[["dense_model", "tokenizer", "method", "lang", "Recall@1", "Recall@10", "MRR@10"]]
      .sort_values(["dense_model", "tokenizer", "method", "lang"]).to_string(index=False))
