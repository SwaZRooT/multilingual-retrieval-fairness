import json, re, sys, time, pathlib
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

N = 100
RAW = pathlib.Path("data/raw"); P = pathlib.Path("data/processed")
KEY = ["link", "question_number"]
en = pd.read_parquet(RAW / "belebele_en.parquet").reset_index(drop=True)
idxs = sorted(np.random.default_rng(0).choice(len(en), N, replace=False).tolist())
NAME = "facebook/nllb-200-distilled-600M"
eng = None
for lang, src in (("hi", "hin_Deva"), ("ml", "mal_Mlym")):
    out_file = P / f"options_{lang}2en.jsonl"
    if out_file.exists():
        print(lang, "already done"); continue
    d = pd.read_parquet(RAW / f"belebele_{lang}.parquet").set_index(KEY)
    df = d.loc[list(zip(en["link"], en["question_number"]))].reset_index()
    tok = AutoTokenizer.from_pretrained(NAME, src_lang=src)
    model = AutoModelForSeq2SeqLM.from_pretrained(NAME).eval()
    eng = tok.convert_tokens_to_ids("eng_Latn")
    t0 = time.time()
    with open(out_file, "w", encoding="utf-8") as f:
        for n, i in enumerate(idxs, 1):
            opts = [str(df.iloc[i][f"mc_answer{k}"]) for k in (1, 2, 3, 4)]
            enc = tok(opts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            with torch.inference_mode():
                gen = model.generate(**enc, forced_bos_token_id=eng, num_beams=1, max_new_tokens=100)
            f.write(json.dumps({"i": i, "opts": tok.batch_decode(gen, skip_special_tokens=True)},
                               ensure_ascii=False) + "\n")
            if n % 20 == 0:
                print(lang, n, f"{(time.time()-t0)/n:.1f} s/question", flush=True)
