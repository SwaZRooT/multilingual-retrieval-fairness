import json, sys, time, pathlib, urllib.request
import numpy as np, pandas as pd

MODEL = sys.argv[1]
N = 100
EMB = "bge-m3"
P = pathlib.Path("data/processed"); RAW = pathlib.Path("data/raw"); RES = pathlib.Path("results")
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
qrels = json.load(open(P / "qrels.json"))
en = pd.read_parquet(RAW / "belebele_en.parquet").reset_index(drop=True)
gold_ans = en["correct_answer_num"].astype(int).values
idxs = sorted(np.random.default_rng(0).choice(len(en), N, replace=False).tolist())

cache_file = RES / f"rag_tr_cache_{MODEL.replace(':', '_')}.jsonl"
cache = {}
if cache_file.exists():
    for l in open(cache_file, encoding="utf-8"):
        r = json.loads(l); cache[r["key"]] = r

def ask(key, passage, question, opts):
    if key in cache:
        return cache[key]["pred"], cache[key]["secs"]
    prompt = ("Read the passage and answer the multiple-choice question. "
              "Reply with only the number (1, 2, 3 or 4) of the correct answer.\n\n"
              f"Passage:\n{passage}\n\nQuestion: {question}\n"
              f"1. {opts[0]}\n2. {opts[1]}\n3. {opts[2]}\n4. {opts[3]}\n\nAnswer:")
    body = json.dumps({"model": MODEL, "prompt": prompt, "stream": False,
                       "options": {"temperature": 0, "num_predict": 4, "num_ctx": 4096}}).encode()
    req = urllib.request.Request("http://localhost:11434/api/generate", body,
                                 {"Content-Type": "application/json"})
    t0 = time.time()
    out = json.loads(urllib.request.urlopen(req, timeout=900).read())["response"]
    secs = time.time() - t0
    pred = next((int(ch) for ch in out if ch in "1234"), 0)
    cache[key] = {"key": key, "pred": pred, "secs": secs}
    with open(cache_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(cache[key]) + "\n")
    return pred, secs

rows = []
for lang in ["hi", "ml"]:
    corpus = read(P / f"corpus_{lang}2en.jsonl")
    queries = read(P / f"queries_{lang}2en.jsonl")
    text_of = {c["id"]: c["text"] for c in corpus}
    opts_of = {r["i"]: r["opts"] for r in read(P / f"options_{lang}2en.jsonl")}
    z = np.load(RES / "emb" / f"{EMB}_{lang}2en.npz")
    top1 = (z["Q"] @ z["D"].T).argmax(1)
    oracle_ok, rag_ok, hits, secs_all = [], [], [], []
    for n, i in enumerate(idxs, 1):
        gold_pid = qrels[f"q{i:04d}"]; ret_pid = corpus[top1[i]]["id"]
        q, o = queries[i]["text"], opts_of[i]
        p_or, s1 = ask(f"tr|{lang}|{i}|{gold_pid}", text_of[gold_pid], q, o)
        if ret_pid == gold_pid:
            p_rag, s2 = p_or, 0.0
        else:
            p_rag, s2 = ask(f"tr|{lang}|{i}|{ret_pid}", text_of[ret_pid], q, o)
        oracle_ok.append(p_or == gold_ans[i]); rag_ok.append(p_rag == gold_ans[i])
        hits.append(ret_pid == gold_pid); secs_all.append(s1 + s2)
        if n % 10 == 0:
            print(f"{lang} {n}/{N} oracle={np.mean(oracle_ok):.3f} rag={np.mean(rag_ok):.3f}", flush=True)
    rows.append({"pipeline": "translate", "lang": lang, "n": N,
                 "retrieval_hit@1": np.mean(hits), "oracle_acc": np.mean(oracle_ok),
                 "rag_acc": np.mean(rag_ok), "avg_llm_sec_per_q": np.mean(secs_all)})

df = pd.DataFrame(rows).round(3)
df.to_csv(RES / "rag_quality_translate.csv", index=False)
print(df.to_string())
