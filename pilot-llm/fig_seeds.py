# -*- coding: utf-8 -*-
"""Figure du pilote multi-graines (idee-24) : delta de perte par condition."""
import json
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import os

WORK = os.path.dirname(os.path.abspath(__file__))
rows = json.load(open(os.path.join(WORK, "results_seeds.json")))
OUT = os.path.normpath(os.path.join(WORK, "..", "figures"))

conds = ["lin_stable", "lin_drift", "log_stable", "log_drift", "log_drift_scaled"]
labels = ["MeZO (lin)\ncapteur stable", "MeZO (lin)\ncapteur dérivant",
          "ColorMeZO\ncapteur stable\n(η égal)", "ColorMeZO\ncapteur dérivant\n(η égal)",
          "ColorMeZO\ncapteur dérivant\n(η×ℓ₀)"]
colors = ["#1d4ed8", "#93c5fd", "#ea580c", "#fdba74", "#dc2626"]

plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linestyle": "--",
    "axes.axisbelow": True, "svg.fonttype": "none",
})

fig, ax = plt.subplots(figsize=(9.0, 4.8))
for i, c in enumerate(conds):
    ds = sorted(r["delta"] for r in rows if r["cond"] == c)
    ax.scatter([i] * len(ds), ds, s=60, color=colors[i], zorder=3,
               edgecolors="white", linewidths=0.8)
    med = st.median(ds)
    ax.hlines(med, i - 0.28, i + 0.28, color="#111827", lw=2.2, zorder=4)
    ax.text(i + 0.12, med - 0.006, f"{med:+.3f}".replace(".", ","),
            fontsize=9, va="top", color="#111827")
ax.axhline(0, color="#9ca3af", lw=1, ls="--")
ax.text(4.42, -0.005, "aucune descente", fontsize=8.5, color="#6b7280", ha="right", va="top")
ax.text(4.42, -0.325, "meilleure descente ↓", fontsize=8.5, color="#6b7280", ha="right")
ax.set_xticks(range(len(conds)))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_xlim(-0.5, 4.5)
ax.set_ylim(-0.335, 0.02)
ax.set_ylabel("Δ perte vraie (finale − initiale, 5 graines)")
ax.set_title("Pilote OPT-125M multi-graines : le mode scalé (η×ℓ₀) rejoint MeZO sous dérive, mais la dérive ne mord pas encore à 30 lectures (5 graines)",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "pilot_seeds.svg"), bbox_inches="tight")
print("OK", os.path.join(OUT, "pilot_seeds.svg"))
