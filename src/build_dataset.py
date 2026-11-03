import json, pathlib
from datasets import load_dataset

LANGS = {"en": "eng_Latn", "hi": "hin_Deva", "ml": "mal_Mlym"}
RAW = pathlib.Path("data/raw"); RAW.mkdir(parents=True, exist_ok=True)
OUT = pathlib.Path("data/processed"); OUT.mkdir(parents=True, exist_ok=True)

def load_lang(cfg):
    # Belebele stores each language as a split; fall back to config just in case
    try:
        return load_dataset("facebook/belebele", split=cfg).to_pandas()
    except Exception:
        return load_dataset("facebook/belebele", cfg, split="test").to_pandas()

dfs = {}
for code, cfg in LANGS.items():
    dfs[code] = load_lang(cfg)
    dfs[code].to_parquet(RAW / f"belebele_{code}.parquet")
    print(code, dfs[code].shape)

# Align every language to the English row order using (link, question_number)
KEY = ["link", "question_number"]
en = dfs["en"].reset_index(drop=True)
for code in ("hi", "ml"):
    d = dfs[code].set_index(KEY)
    dfs[code] = d.loc[list(zip(en["link"], en["question_number"]))].reset_index()
    assert (dfs[code]["link"].values == en["link"].values).all()
    assert (dfs[code]["question_number"].values == en["question_number"].values).all()
dfs["en"] = en
print("Row alignment OK")

# Passage IDs come from the English passage text; reused for all languages by row
uniq = list(en["flores_passage"].unique())
pid_of = {p: f"p{i:03d}" for i, p in enumerate(uniq)}
row_pid = en["flores_passage"].map(pid_of).tolist()

qrels = {f"q{i:04d}": row_pid[i] for i in range(len(en))}
json.dump(qrels, open(OUT / "qrels.json", "w"), indent=1)

for code, df in dfs.items():
    # corpus: one entry per unique passage, taken from this language's own column
    seen, corpus = set(), []
    for i, pid in enumerate(row_pid):
        if pid not in seen:
            seen.add(pid)
            corpus.append({"id": pid, "text": df["flores_passage"].iloc[i]})
    queries = [{"id": f"q{i:04d}", "text": df["question"].iloc[i]} for i in range(len(df))]
    for name, rows in (("corpus", corpus), ("queries", queries)):
        with open(OUT / f"{name}_{code}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(code, "corpus:", len(corpus), "queries:", len(queries))
