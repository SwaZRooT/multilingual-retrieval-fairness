import json, time, random, pathlib
import numpy as np, pandas as pd, torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

P = pathlib.Path("data/processed"); RES = pathlib.Path("results")
LANGS = ["en", "hi", "ml"]
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
qrels = json.load(open(P / "qrels.json"))
random.seed(0); torch.manual_seed(0)

pids = sorted({v for v in qrels.values()})
random.Random(0).shuffle(pids)
test_p = set(pids[: int(0.3 * len(pids))])          # 30% of passages held out
print("train passages:", len(pids) - len(test_p), "test passages:", len(test_p))

data = {l: (read(P / f"corpus_{l}.jsonl"), read(P / f"queries_{l}.jsonl")) for l in LANGS}

def build_train():
    ex = []
    for l in LANGS:
        corpus, queries = data[l]
        text = {c["id"]: c["text"] for c in corpus}
        for q in queries:
            p = qrels[q["id"]]
            if p not in test_p:
                ex.append(InputExample(texts=["query: " + q["text"], "passage: " + text[p]]))
    return ex

def evaluate(model, tag):
    rows = []
    for l in LANGS:
        corpus, queries = data[l]
        cids = [c["id"] for c in corpus if c["id"] in test_p]
        ctext = [c["text"] for c in corpus if c["id"] in test_p]
        tq = [q for q in queries if qrels[q["id"]] in test_p]
        D = model.encode(["passage: " + t for t in ctext], batch_size=8, normalize_embeddings=True)
        Q = model.encode(["query: " + q["text"] for q in tq], batch_size=32, normalize_embeddings=True)
        idx = {c: i for i, c in enumerate(cids)}
        gold = np.array([idx[qrels[q["id"]]] for q in tq])
        sims = Q @ D.T
        rank = (sims > sims[np.arange(len(gold)), gold][:, None]).sum(1) + 1
        rows.append({"model": tag, "lang": l, "n_queries": len(tq), "n_passages": len(cids),
                     "Recall@1": (rank <= 1).mean(), "Recall@5": (rank <= 5).mean(),
                     "MRR@10": np.where(rank <= 10, 1 / rank, 0).mean()})
    return rows

model = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
model.max_seq_length = 256
rows = evaluate(model, "e5-small base")

train = build_train()
print("training pairs:", len(train))
loader = DataLoader(train, shuffle=True, batch_size=16)
loss = losses.MultipleNegativesRankingLoss(model)
t0 = time.time()
model.fit(train_objectives=[(loader, loss)], epochs=1, warmup_steps=20,
          optimizer_params={"lr": 2e-5}, show_progress_bar=True)
train_secs = round(time.time() - t0, 1)
print("training seconds:", train_secs)
rows += evaluate(model, "e5-small finetuned")
for r in rows:
    r["train_seconds"] = train_secs if "finetuned" in r["model"] else 0

df = pd.DataFrame(rows)
df.to_csv(RES / "finetune.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(3).to_string(index=False))
