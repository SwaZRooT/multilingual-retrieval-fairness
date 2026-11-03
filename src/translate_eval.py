import json, sys, time, pathlib
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

P = pathlib.Path("data/processed"); RES = pathlib.Path("results"); EMB = RES / "emb"
MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "bge-m3":   ("BAAI/bge-m3", "", ""),
}
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
qrels = json.load(open(P / "qrels.json"))
PUNCT = ".,;:!?\"'()[]{}“”‘’-–—"
tok = lambda t: [w for w in (x.strip(PUNCT) for x in t.lower().split()) if w]
zscore = lambda s: (s - s.mean(1, keepdims=True)) / (s.std(1, keepdims=True) + 1e-9)

def metrics(sims, gold):
    rank = (sims > sims[np.arange(len(gold)), gold][:, None]).sum(1) + 1
    return {"Recall@1": (rank <= 1).mean(), "Recall@5": (rank <= 5).mean(),
            "Recall@10": (rank <= 10).mean(), "MRR@10": np.where(rank <= 10, 1 / rank, 0).mean()}

key = sys.argv[1]
name, qp, dp = MODELS[key]
model = SentenceTransformer(name, device="cpu"); model.max_seq_length = 512
rows = []
for lang in ["hi", "ml"]:
    corpus = read(P / f"corpus_{lang}2en.jsonl"); queries = read(P / f"queries_{lang}2en.jsonl")
    assert len(corpus) == 488 and len(queries) == 900, (len(corpus), len(queries))
    f = EMB / f"{key}_{lang}2en.npz"
    secs = None
    if f.exists():
        z = np.load(f); D, Q = z["D"], z["Q"]
    else:
        t0 = time.time()
        D = model.encode([dp + c["text"] for c in corpus], batch_size=8, normalize_embeddings=True)
        Q = model.encode([qp + q["text"] for q in queries], batch_size=32, normalize_embeddings=True)
        secs = round(time.time() - t0, 1)
        np.savez(f, D=D, Q=Q)
    idx = {c["id"]: i for i, c in enumerate(corpus)}
    gold = np.array([idx[qrels[q["id"]]] for q in queries])
    dense = Q @ D.T
    bm = BM25Okapi([tok(c["text"]) for c in corpus])
    sparse = np.array([bm.get_scores(tok(q["text"])) for q in queries])
    variants = {"dense": dense, "bm25": sparse,
                "hybrid_w0.5": 0.5 * zscore(dense) + 0.5 * zscore(sparse),
                "hybrid_w0.7": 0.7 * zscore(dense) + 0.3 * zscore(sparse)}
    for vname, s in variants.items():
        m = metrics(s, gold)
        m.update(model=key, lang=lang, method="translate+" + vname, encode_seconds=secs)
        rows.append(m)

df = pd.DataFrame(rows)
out = RES / "translation.csv"
if out.exists():
    df = pd.concat([pd.read_csv(out), df]).drop_duplicates(["model", "lang", "method"], keep="last")
df.to_csv(out, index=False)
pd.set_option("display.width", 200)
print(df[df.model == key].round(3)[["model", "lang", "method", "Recall@1", "Recall@5", "Recall@10", "MRR@10"]].to_string(index=False))
