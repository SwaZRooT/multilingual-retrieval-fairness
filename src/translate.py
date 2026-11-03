import json, re, sys, time, pathlib
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

lang, kind = sys.argv[1], sys.argv[2]          # hi|ml  and  corpus|queries
limit = int(sys.argv[3]) if len(sys.argv) > 3 else None
SRC = {"hi": "hin_Deva", "ml": "mal_Mlym"}[lang]
P = pathlib.Path("data/processed")
out_file = P / f"{kind}_{lang}2en.jsonl"
NAME = "facebook/nllb-200-distilled-600M"

items = [json.loads(l) for l in open(P / f"{kind}_{lang}.jsonl", encoding="utf-8")]
done = {}
if out_file.exists():
    for l in open(out_file, encoding="utf-8"):
        r = json.loads(l); done[r["id"]] = r
todo = [it for it in items if it["id"] not in done]
if limit:
    todo = todo[:limit]
print(f"{lang} {kind}: {len(done)} done, {len(todo)} to translate", flush=True)

tok = AutoTokenizer.from_pretrained(NAME, src_lang=SRC)
model = AutoModelForSeq2SeqLM.from_pretrained(NAME).eval()
eng = tok.convert_tokens_to_ids("eng_Latn")

def translate(text):
    sents = [s for s in re.split(r"(?<=[।.?!])\s+", text.strip()) if s]
    outs = []
    for i in range(0, len(sents), 8):
        enc = tok(sents[i:i + 8], return_tensors="pt", padding=True, truncation=True, max_length=200)
        with torch.inference_mode():
            gen = model.generate(**enc, forced_bos_token_id=eng, num_beams=1, max_new_tokens=200)
        outs += tok.batch_decode(gen, skip_special_tokens=True)
    return " ".join(outs)

t0 = time.time()
with open(out_file, "a", encoding="utf-8") as f:
    for n, it in enumerate(todo, 1):
        f.write(json.dumps({"id": it["id"], "text": translate(it["text"])}, ensure_ascii=False) + "\n")
        f.flush()
        if n % 10 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}  {(time.time()-t0)/n:.1f} s/item", flush=True)
print("total seconds:", round(time.time() - t0, 1))
