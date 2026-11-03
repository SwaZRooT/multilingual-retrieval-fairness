import json, pathlib
import numpy as np, pandas as pd

P = pathlib.Path("data/processed"); RAW = pathlib.Path("data/raw"); RES = pathlib.Path("results")
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
qrels = json.load(open(P / "qrels.json"))
en = pd.read_parquet(RAW / "belebele_en.parquet").reset_index(drop=True)
gold = en["correct_answer_num"].astype(int).values
idxs = sorted(np.random.default_rng(0).choice(len(en), 100, replace=False).tolist())
rng = np.random.default_rng(1)

def load(path):
    d = {}
    for l in open(path, encoding="utf-8"):
        r = json.loads(l); d[r["key"]] = r["pred"]
    return d

def correct(llm, lang, translated):
    tag = llm.replace(":", "_")
    cache = load(RES / (f"rag_tr_cache_{tag}.jsonl" if translated else f"rag_cache_{tag}.jsonl"))
    suffix = "2en" if translated else ""
    corpus = read(P / f"corpus_{lang}{suffix}.jsonl")
    z = np.load(RES / "emb" / f"bge-m3_{lang}{suffix}.npz")
    top1 = (z["Q"] @ z["D"].T).argmax(1)
    out = []
    for i in idxs:
        ret = corpus[top1[i]]["id"]
        key = f"tr|{lang}|{i}|{ret}" if translated else f"{lang}|{i}|{ret}"
        out.append(cache[key] == gold[i])
    return np.array(out, dtype=float)

def ci(x):
    b = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(5000)]
    return np.percentile(b, [2.5, 97.5])

rows = []
for llm in ["qwen2.5:3b", "qwen2.5:7b"]:
    res = {("en", False): correct(llm, "en", False)}
    for lang in ["hi", "ml"]:
        res[(lang, False)] = correct(llm, lang, False)
        res[(lang, True)] = correct(llm, lang, True)
    for (lang, tr), x in res.items():
        lo, hi = ci(x)
        rows.append({"llm": llm, "lang": lang, "pipeline": "translate" if tr else "native",
                     "rag_acc": x.mean(), "ci_lo": lo, "ci_hi": hi})
    for lang in ["hi", "ml"]:
        d = res[(lang, True)] - res[(lang, False)]
        lo, hi = ci(d)
        rows.append({"llm": llm, "lang": lang, "pipeline": "translate minus native (paired)",
                     "rag_acc": d.mean(), "ci_lo": lo, "ci_hi": hi})
df = pd.DataFrame(rows).round(3)
df.to_csv(RES / "answer_ci.csv", index=False)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
