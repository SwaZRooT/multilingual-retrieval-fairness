import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results/baseline.csv")
METRICS = ["Recall@1", "Recall@5", "Recall@10", "MRR@10"]

# Gap table: absolute and relative drop vs English
rows = []
for model, g in df.groupby("model"):
    en = g[g.lang == "en"].iloc[0]
    for _, r in g[g.lang != "en"].iterrows():
        row = {"model": model, "lang": r.lang}
        for m in METRICS:
            row[f"{m} gap"] = round(en[m] - r[m], 3)
            row[f"{m} rel%"] = round(100 * (en[m] - r[m]) / en[m], 1)
        rows.append(row)
gap = pd.DataFrame(rows)
gap.to_csv("results/language_gap.csv", index=False)
pd.set_option("display.width", 250)
print(gap)

# Fairness ratio: worst language / best language (1.0 = perfectly fair)
fair = df.pivot(index="model", columns="lang", values="Recall@1")
fair["fairness_ratio"] = (fair.min(axis=1) / fair.max(axis=1)).round(3)
print("\nFairness ratio (min/max Recall@1):\n", fair)
fair.to_csv("results/fairness_ratio.csv")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, metric in zip(axes, ["Recall@1", "MRR@10"]):
    df.pivot(index="model", columns="lang", values=metric)[["en", "hi", "ml"]].plot.bar(ax=ax)
    ax.set_title(metric); ax.set_ylim(0, 1); ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.savefig("results/language_gap.png", dpi=150)
print("\nSaved results/language_gap.csv, fairness_ratio.csv, language_gap.png")
