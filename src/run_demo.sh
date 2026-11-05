#!/usr/bin/env bash
# Silent demo: prints a caption, then the real result, then waits for Enter.
cd "$(dirname "$0")/.." || exit 1
[ -f .venv/bin/activate ] && source .venv/bin/activate

banner() { clear; echo; echo "=============================================================="; echo "  $1"; [ -n "$2" ] && echo "  $2"; echo "=============================================================="; echo; }
pause()  { echo; read -r -p "  [press Enter for next step] " _; }

banner "MULTILINGUAL RETRIEVAL FAIRNESS  (English / Hindi / Malayalam)" \
       "Does a public-services assistant work equally well in all three languages?"
echo "  What was built:"
echo "   - parallel dataset: same 488 passages + 900 questions in EN, HI, ML (Belebele)"
echo "   - retrieval: 3 embedding models + keyword (BM25) + hybrid search"
echo "   - answer quality: local Qwen2.5 LLM, oracle vs retrieved passage"
echo "   - fixes tested: hybrid search, translation to English, fine-tuning"
pause

banner "STEP 1: THE DATA" "Same content in three languages, so any gap comes from language or model"
echo "Lines per file (488 passages, 900 questions):"
wc -l data/processed/corpus_en.jsonl data/processed/corpus_hi.jsonl data/processed/corpus_ml.jsonl \
      data/processed/queries_en.jsonl data/processed/queries_hi.jsonl data/processed/queries_ml.jsonl
pause

banner "STEP 2: RETRIEVAL GAP (Recall@1)" "How often the correct passage is ranked first. Fairness ratio = worst / best language"
python - << 'PY'
import pandas as pd
b = pd.read_csv("results/baseline.csv")
t = b.pivot(index="model", columns="lang", values="Recall@1")[["en", "hi", "ml"]]
t["fairness_ratio"] = pd.read_csv("results/fairness_ratio.csv").set_index("model")["fairness_ratio"]
print(t.round(3).to_string())
PY
echo
echo "  -> Every model is worst on a non-English language. MiniLM nearly fails on Malayalam."
pause

banner "STEP 3: LIVE RETRIEVAL CASES (bge-m3)" "Real questions found in English but missed in Hindi / Malayalam"
python src/demo.py
echo
echo "  -> Next: switch to the browser tab 'Retrieval demo' (results/demo.html)"
pause

banner "STEP 4: ANSWER QUALITY (Qwen2.5, 100 questions per language)" "RAG accuracy with 95% bootstrap intervals; random guessing = 0.25"
python - << 'PY'
import pandas as pd
a = pd.read_csv("results/answer_ci.csv")
a["95% CI"] = a.ci_lo.round(2).astype(str) + " to " + a.ci_hi.round(2).astype(str)
print(a[["llm", "lang", "pipeline", "rag_acc", "95% CI"]].to_string(index=False))
PY
echo
echo "  -> KEY FINDING: Malayalam retrieval is fine, but the LLM answers far worse (generator is the weak link)."
pause

banner "STEP 5: MITIGATIONS (Recall@1)" "Hybrid search vs translation to English, compared with baseline"
python - << 'PY'
import pandas as pd
pd.set_option("display.width", 200)
m = pd.read_csv("results/mitigation_comparison.csv")
p = m.pivot_table(index=["model", "lang"], columns="method", values="Recall@1")
order = ["baseline dense", "hybrid w0.7 (4-gram BM25)", "translate + dense", "translate + hybrid w0.7"]
print(p[[c for c in order if c in p.columns]].round(3).to_string())
PY
echo
echo "  -> Hybrid search (seconds of CPU) closes most of the Malayalam retrieval gap."
pause

banner "STEP 6: COST OF EACH STEP" "Single laptop CPU"
python - << 'PY'
import pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 80)
print(pd.read_csv("results/cost_table.csv").to_string(index=False))
PY
echo
echo "  -> Translation helps Malayalam answers but costs ~1 hour per language. Fine-tuning: no measurable gain."
pause

banner "LIMITATIONS AND NEXT STEPS"
echo "  - answer quality uses 100 questions per language (wide intervals)"
echo "  - one LLM family; no distributed-scale experiments"
echo "  - next: stronger multilingual LLM, larger sample, tune fusion weight on held-out split"
echo
echo "  Code, report, notebook and data card: github.com/SwaZRooT/multilingual-retrieval-fairness"
echo
