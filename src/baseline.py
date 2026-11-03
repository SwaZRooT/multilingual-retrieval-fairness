import json, sys, time, pathlib
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer

LANGS = ["en", "hi", "ml"]
P = pathlib.Path("data/processed")
RES = pathlib.Path("results"); RES.mkdir(exist_ok=True)

MODELS = {
    "e5-small": ("intfloat/multilingual-e5-small", "query: ", "passage: "),
    "minilm":   ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    "bge-m3":   ("BAAI/bge-m3", "", ""),
}

def read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]

qrels = json.load(open(P / "qrels.json"))

def evaluate(qids, pids, Q, D):
    sims = Q @ D.T
    pid_idx = {p: i for i, p in enumerate(pids)}
    gold = np.array([pid_idx[qrels[q]] for q in qids])
    gold_sim = sims[np.arange(len(qids)), gold][:, None]
    rank = (sims > gold_sim).sum(axis=1) + 1   # 1 = best
    return {
        "Recall@1": float((rank <= 1).mean()),
        "Recall@5": float((rank <= 5).mean()),
        "Recall@10": float((rank <= 10).mean()),
        "MRR@10": float(np.where(rank <= 10, 1.0 / rank, 0).mean()),
    }

rows = []
for key in sys.argv[1:]:
    name, qp, dp = MODELS[key]
    print(f"\n=== {key} ({name}) ===")
    model = SentenceTransformer(name, device="cpu")
    model.max_seq_length = 512
    for lang in LANGS:
        corpus = read(P / f"corpus_{lang}.jsonl")
        queries = read(P / f"queries_{lang}.jsonl")
        t0 = time.time()
        D = model.encode([dp + c["text"] for c in corpus], batch_size=8,
                         normalize_embeddings=True, show_progress_bar=True)
        Q = model.encode([qp + q["text"] for q in queries], batch_size=32,
                         normalize_embeddings=True, show_progress_bar=False)
        secs = time.time() - t0
        m = evaluate([q["id"] for q in queries], [c["id"] for c in corpus], Q, D)
        m.update(model=key, lang=lang, encode_seconds=round(secs, 1))
        rows.append(m)
        print(lang, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in m.items()})

out = RES / "baseline.csv"
df = pd.DataFrame(rows)
if out.exists():
    df = pd.concat([pd.read_csv(out), df]).drop_duplicates(["model", "lang"], keep="last")
df.to_csv(out, index=False)
print("\nSaved", out)
