"""Live demo: shows real retrieval cases from saved bge-m3 embeddings (no model loading, runs in ~2 s).
Writes results/demo.html (browser renders Hindi/Malayalam correctly; terminals often do not)."""
import json, html, pathlib
import numpy as np

P = pathlib.Path("data/processed"); R = pathlib.Path("results")
read = lambda p: [json.loads(l) for l in open(p, encoding="utf-8")]
qrels = json.load(open(P / "qrels.json"))
LANGS = ["en", "hi", "ml"]
NAME = {"en": "English", "hi": "Hindi", "ml": "Malayalam"}

D = {}
for l in LANGS:
    corpus, queries = read(P / f"corpus_{l}.jsonl"), read(P / f"queries_{l}.jsonl")
    z = np.load(R / "emb" / f"bge-m3_{l}.npz")
    D[l] = dict(corpus=corpus, queries=queries, top=(z["Q"] @ z["D"].T).argmax(1))

print("Overall Recall@1 with bge-m3 (all 900 questions):")
hit = {}
for l in LANGS:
    ok = np.array([D[l]["corpus"][D[l]["top"][i]]["id"] == qrels[D[l]["queries"][i]["id"]]
                   for i in range(len(D[l]["queries"]))])
    hit[l] = ok
    print(f"  {NAME[l]:10s} {ok.mean():.3f}")

n = len(D["en"]["queries"])
all_ok = [i for i in range(n) if all(hit[l][i] for l in LANGS)]
ml_fail = [i for i in range(n) if hit["en"][i] and not hit["ml"][i]]
hi_fail = [i for i in range(n) if hit["en"][i] and not hit["hi"][i]]
cases = [("Works in all three languages", all_ok[:1]),
         ("Found in English but missed in Malayalam", ml_fail[:2]),
         ("Found in English but missed in Hindi", hi_fail[:1])]
print(f"\nQuestions found in English but missed in Malayalam: {len(ml_fail)} of {n}")
print(f"Questions found in English but missed in Hindi:     {len(hi_fail)} of {n}")

def snippet(t, k=230): return html.escape(t[:k]) + ("..." if len(t) > k else "")
out = ["<html><head><meta charset='utf-8'><title>Retrieval demo</title><style>",
       "body{font-family:'Noto Sans','Noto Sans Devanagari','Noto Sans Malayalam',sans-serif;margin:24px;max-width:1100px}",
       "table{border-collapse:collapse;width:100%;margin-bottom:28px}td,th{border:1px solid #999;padding:8px;vertical-align:top}",
       "th{background:#dbeafe}.ok{color:#166534;font-weight:bold}.bad{color:#b91c1c;font-weight:bold}",
       "h2{color:#1e3a8a}</style></head><body><h1>Same question, three languages (bge-m3, top-1 retrieval)</h1>"]
for title, idxs in cases:
    for i in idxs:
        gold = qrels[D["en"]["queries"][i]["id"]]
        out.append(f"<h2>{html.escape(title)}: question {i}</h2><table><tr><th>Language</th><th>Question</th><th>Retrieved passage</th><th>Result</th></tr>")
        for l in LANGS:
            ret = D[l]["corpus"][D[l]["top"][i]]
            good = ret["id"] == gold
            out.append(f"<tr><td>{NAME[l]}</td><td>{html.escape(D[l]['queries'][i]['text'])}</td>"
                       f"<td><b>{ret['id']}</b><br>{snippet(ret['text'])}</td>"
                       f"<td class='{'ok' if good else 'bad'}'>{'correct' if good else 'wrong (correct is ' + gold + ')'}</td></tr>")
        out.append("</table>")
out.append("</body></html>")
(R / "demo.html").write_text("\n".join(out), encoding="utf-8")
print("\nWrote results/demo.html  ->  open it with:  xdg-open results/demo.html")
