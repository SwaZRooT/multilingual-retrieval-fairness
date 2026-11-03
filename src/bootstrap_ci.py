import json, sys, pathlib
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

P = pathlib.Path("data/processed"); EMB = pathlib.Path("results/emb"); EMB.mkdir(parents=True, exist_ok=True)
LANGS = ["en", "hi", "ml"]
MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "bge-m3":   ("BAAI/bge-m3", "", ""),
}
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
qrels = json.load(open(P / "qrels.json"))
rng = np.random.default_rng(0)

key = sys.argv[1]
name, qp, dp = MODELS[key]
model = SentenceTransformer(name, device="cpu"); model.max_seq_length = 512
rows = []
for lang in LANGS:
    f = EMB / f"{key}_{lang}.npz"
    corpus, queries = read(P / f"corpus_{lang}.jsonl"), read(P / f"queries_{lang}.jsonl")
    if f.exists():
        z = np.load(f); D, Q = z["D"], z["Q"]
    else:
        D = model.encode([dp + c["text"] for c in corpus], batch_size=8, normalize_embeddings=True)
        Q = model.encode([qp + q["text"] for q in queries], batch_size=32, normalize_embeddings=True)
        np.savez(f, D=D, Q=Q)
    idx = {c["id"]: i for i, c in enumerate(corpus)}
    gold = np.array([idx[qrels[q["id"]]] for q in queries])
    sims = Q @ D.T
    rank = (sims > sims[np.arange(len(gold)), gold][:, None]).sum(1) + 1
    hit = (rank <= 1).astype(float)
    boots = [hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    rows.append({"model": key, "lang": lang, "Recall@1": hit.mean(), "ci_lo": lo, "ci_hi": hi})
    np.save(EMB / f"{key}_{lang}_hit.npy", hit)
df = pd.DataFrame(rows).round(3)
print(df)
out = pathlib.Path("results/bootstrap_ci.csv")
if out.exists():
    df = pd.concat([pd.read_csv(out), df]).drop_duplicates(["model", "lang"], keep="last")
df.to_csv(out, index=False)
