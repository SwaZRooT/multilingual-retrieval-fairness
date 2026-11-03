import json, sys, time, pathlib, urllib.request
import numpy as np, pandas as pd

MODEL = sys.argv[1]
N = int(sys.argv[2])
EMB = "bge-m3"
LANGS = ["en", "hi", "ml"]
P = pathlib.Path("data/processed"); RAW = pathlib.Path("data/raw"); RES = pathlib.Path("results")
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
qrels = json.load(open(P / "qrels.json"))

# Re-align each language to the English row order (raw parquet is not aligned)
KEY = ["link", "question_number"]
en = pd.read_parquet(RAW / "belebele_en.parquet").reset_index(drop=True)
def aligned(code):
    if code == "en":
        return en
    d = pd.read_parquet(RAW / f"belebele_{code}.parquet").set_index(KEY)
    return d.loc[list(zip(en["link"], en["question_number"]))].reset_index()
dfs = {c: aligned(c) for c in LANGS}
gold_ans = en["correct_answer_num"].astype(int).values

# Same random sample of questions for every language
idxs = sorted(np.random.default_rng(0).choice(len(en), N, replace=False).tolist())

cache_file = RES / f"rag_cache_{MODEL.replace(':', '_').replace('/', '_')}.jsonl"
cache = {}
if cache_file.exists():
    for l in open(cache_file, encoding="utf-8"):
        r = json.loads(l); cache[r["key"]] = r

def ask(lang, i, pid, passage):
    key = f"{lang}|{i}|{pid}"
    if key in cache:
        return cache[key]["pred"], cache[key]["secs"]
    r = dfs[lang].iloc[i]
    prompt = (
        "Read the passage and answer the multiple-choice question. "
        "Reply with only the number (1, 2, 3 or 4) of the correct answer.\n\n"
        f"Passage:\n{passage}\n\nQuestion: {r['question']}\n"
        f"1. {r['mc_answer1']}\n2. {r['mc_answer2']}\n3. {r['mc_answer3']}\n4. {r['mc_answer4']}\n\nAnswer:"
    )
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 4, "num_ctx": 4096}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())["response"]
    secs = time.time() - t0
    pred = next((int(ch) for ch in out if ch in "1234"), 0)   # 0 = unparseable
    cache[key] = {"key": key, "pred": pred, "secs": secs}
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(cache[key]) + "\n")
    return pred, secs

rows = []
for lang in LANGS:
    corpus = read(P / f"corpus_{lang}.jsonl")
    text_of = {c["id"]: c["text"] for c in corpus}
    z = np.load(RES / "emb" / f"{EMB}_{lang}.npz")
    top1 = (z["Q"] @ z["D"].T).argmax(1)
    oracle_ok, rag_ok, hits, secs_all = [], [], [], []
    for n, i in enumerate(idxs):
        gold_pid = qrels[f"q{i:04d}"]
        ret_pid = corpus[top1[i]]["id"]
        p_or, s1 = ask(lang, i, gold_pid, text_of[gold_pid])
        if ret_pid == gold_pid:
            p_rag, s2 = p_or, 0.0
        else:
            p_rag, s2 = ask(lang, i, ret_pid, text_of[ret_pid])
        oracle_ok.append(p_or == gold_ans[i]); rag_ok.append(p_rag == gold_ans[i])
        hits.append(ret_pid == gold_pid); secs_all.append(s1 + s2)
        if (n + 1) % 10 == 0:
            print(f"{lang} {n+1}/{N}  oracle={np.mean(oracle_ok):.3f}  rag={np.mean(rag_ok):.3f}", flush=True)
    rows.append({"llm": MODEL, "lang": lang, "n": N, "retrieval_hit@1": np.mean(hits),
                 "oracle_acc": np.mean(oracle_ok), "rag_acc": np.mean(rag_ok),
                 "avg_sec_per_q": np.mean(secs_all)})

df = pd.DataFrame(rows).round(3)
print(df.to_string())
df.to_csv(RES / f"rag_quality_{MODEL.replace(':', '_').replace('/', '_')}.csv", index=False)
