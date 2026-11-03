import pandas as pd
pd.set_option("display.width", 220)
R = "results/"
base = pd.read_csv(R + "baseline.csv")
hyb = pd.read_csv(R + "hybrid.csv")
tr = pd.read_csv(R + "translation.csv")

def b(model, lang, col="Recall@1"):
    return base[(base.model == model) & (base.lang == lang)][col].iloc[0]
def h(model, lang, method):
    d = hyb[(hyb.dense_model == model) & (hyb.tokenizer == "ngram4") & (hyb.method == method) & (hyb.lang == lang)]
    return d["Recall@1"].iloc[0]
def t(model, lang, method):
    d = tr[(tr.model == model) & (tr.lang == lang) & (tr.method == method)]
    return d["Recall@1"].iloc[0]

rows = []
for m in ["e5-small", "bge-m3"]:
    en = b(m, "en")
    for lang in ["hi", "ml"]:
        for name, val in [("baseline dense", b(m, lang)),
                          ("hybrid w0.7 (4-gram BM25)", h(m, lang, "hybrid_w0.7")),
                          ("translate + dense", t(m, lang, "translate+dense")),
                          ("translate + hybrid w0.7", t(m, lang, "translate+hybrid_w0.7"))]:
            rows.append({"model": m, "lang": lang, "method": name,
                         "Recall@1": round(val, 3), "gap_to_en_dense": round(en - val, 3)})
out = pd.DataFrame(rows)
out.to_csv(R + "mitigation_comparison.csv", index=False)
print(out.to_string(index=False))

cost = pd.DataFrame([
    ["e5-small encode (3 langs)", "~55 s", "baseline, small model"],
    ["bge-m3 encode (3 langs)", "~620 s", "about 10x e5-small"],
    ["BM25 / hybrid", "seconds", "no model, no GPU"],
    ["Translate queries (900, per lang)", "~1,400 s", "about 1.5 s/query"],
    ["Translate corpus hi (488)", "3,277 s", "about 6.8 s/passage"],
    ["Translate corpus ml (488)", "~3,500 s est.", "log inflated by laptop suspend; pre-stall rate 7.3 s/passage"],
    ["Fine-tune e5-small (1 epoch)", "663 s", "no measurable gain"],
    ["LLM 3b/7b per question, native ml", "10.6 s / 23.9 s", "long Malayalam prompts"],
    ["LLM 3b/7b per question, translated ml", "1.6 s / 3.6 s", "plus translation cost"],
], columns=["step", "CPU time", "note"])
cost.to_csv(R + "cost_table.csv", index=False)
print("\n", cost.to_string(index=False))
