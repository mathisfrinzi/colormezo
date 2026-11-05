"""
Figures du rapport de l'idée 24 (schéma de principe + campagne LLM SST-2).

Lit uniquement les résultats enregistrés :
  campagne-llm/results/campagne1_ecretage.json   (pas bornés en norme à 0,1)
  campagne-llm/results/campagne2.json            (sans borne, η de 1e-8 à 3e-6)
  campagne-llm/results/campagne3_grille_etendue.json  (la même, grille prolongée à 1e-4)
  campagne-llm/results/campagne4_finale_5graines.json (finale à η réglé sur la précision, 5 graines)
  results/bench_phase_50graines.json             (banc phase sur le jouet, partie 7)
et écrit dans figures/. Quelques secondes, CPU, NumPy + Matplotlib.

La trajectoire du gain ln g(t) n'est pas enregistrée : elle est reconstituée
exactement depuis la graine du capteur (np.random.default_rng(seed + 777),
un tirage N(0, 0,05²) par lecture), comme dans le carnet. Contrôle : son
maximum |ln g| égale le champ ln_g_max enregistré (vérifié à l'exécution).

Usage : python3 src/figures_rapport.py
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "campagne-llm", "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# palette (validée, ordre catégoriel fixe) : MeZO = bleu, ColorMeZO = orange
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
COL = {"oracle": INK, "mezo_stable": BLUE, "mezo_drift": BLUE,
       "color_stable": ORANGE, "color_drift": ORANGE}
LS = {"oracle": ":", "mezo_stable": "--", "color_stable": "--",
      "mezo_drift": "-", "color_drift": "-"}
LAB = {"oracle": "MeZO oracle (perte exacte)", "mezo_stable": "MeZO, capteur stable",
       "color_stable": "ColorMeZO, capteur stable", "mezo_drift": "MeZO, capteur dérivant",
       "color_drift": "ColorMeZO, capteur dérivant"}
METHODS = ["oracle", "mezo_stable", "color_stable", "mezo_drift", "color_drift"]
SIGMA_G = 0.05

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "lines.linewidth": 2, "legend.frameon": False,
    "svg.fonttype": "path", "figure.facecolor": "white",
})


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def runs(R, phase):
    out = {}
    for k, v in R.items():
        _, ph, cfg, eta, seed, reads = k.split("|")
        if ph == phase:
            out[(cfg, eta, int(seed))] = v
    return out


def lng_path(seed, reads):
    rng = np.random.default_rng(seed + 777)
    return np.cumsum(rng.normal(0, SIGMA_G, reads))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    if os.environ.get("APERCU_PNG"):          # copie PNG pour relecture, hors dépôt
        fig.savefig(os.path.join(os.environ["APERCU_PNG"], name.replace(".svg", ".png")),
                    bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("écrit figures/" + name)


C1, C2, C3 = load("campagne1_ecretage.json"), load("campagne2.json"), load("campagne3_grille_etendue.json")
F2, T1, T2 = runs(C2, "final"), runs(C1, "tune"), runs(C2, "tune")
T3, F3 = runs(C3, "tune"), runs(C3, "final")      # campagne 2 + grille prolongée jusqu'à 1e-4
SEEDS = sorted({s for (_, _, s) in F2})
ETA_F = {cfg: e for (cfg, e, _) in F2}

for (cfg, e, s), v in F2.items():           # contrôle de la reconstitution du gain
    if cfg.endswith("drift"):
        assert abs(np.abs(lng_path(s, v["reads"])).max() - v["ln_g_max"]) < 1e-9, (cfg, s)


# ------------------------------------------------------------ 1. schéma de principe
def box(ax, x, y, w, h, text, fc="#f4f3f0", ec=MUTED, fs=9.5, color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color,
            weight=weight, linespacing=1.35)


def arrow(ax, x0, y0, x1, y1, color=MUTED):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12,
                                 lw=1.3, color=color, shrinkA=2, shrinkB=2))


fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
box(ax, 0.1, 1.85, 1.6, 0.9, "poids\nθ ± εz", fs=10)
arrow(ax, 1.7, 2.3, 2.15, 2.3)
box(ax, 2.15, 1.85, 1.6, 0.9, "perte vraie\nℓ₊ , ℓ₋", fs=10)
arrow(ax, 3.75, 2.3, 4.2, 2.3)
box(ax, 4.2, 1.45, 2.3, 1.7,
    "capteur couleur\nu = log₁₀(g · ℓ)\nquantifié 12 bits\nln g dérive à chaque\nlecture (σ_g = 0,05)",
    fc="#fdf6e8", ec="#c98500", fs=9)
ax.text(5.35, 3.35, "on ne lit jamais ℓ,\nseulement u₊ et u₋", ha="center", fontsize=8.5,
        color=MUTED, style="italic")
arrow(ax, 6.5, 2.65, 7.05, 3.35, BLUE)
arrow(ax, 6.5, 1.95, 7.05, 1.25, ORANGE)
box(ax, 7.05, 2.85, 3.8, 1.35,
    "MeZO : reconstruit les valeurs\n10^u₊ − 10^u₋  ≈  g · (ℓ₊ − ℓ₋)\n→ le gain g multiplie le pas",
    fc="#eaf2fc", ec=BLUE, fs=9.5)
box(ax, 7.05, 0.4, 3.8, 1.35,
    "ColorMeZO : différence de logarithmes\nln 10 · (u₊ − u₋)  =  ln ℓ₊ − ln ℓ₋\n→ le gain s'élimine de la paire",
    fc="#fdeee7", ec=ORANGE, fs=9.5)
ax.text(5.5, 0.05, "mise à jour SPSA :  θ ← θ − η · (différence / 2ε) · z      (même z, même graine, mêmes lectures)",
        ha="center", fontsize=8.5, color=MUTED)
ax.set_title("Une paire de lectures, deux façons de la lire", loc="left", fontsize=12, color=INK)
save(fig, "schema_principe.svg")


# ------------------------------------------------------------ 2. schéma du pas effectif
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
lg = lng_path(0, 2000)
axs[0].plot(np.arange(1, 2001), np.exp(lg), color=BLUE)
axs[0].axhline(1, color=MUTED, lw=1, ls="--")
axs[0].set_yscale("log"); axs[0].set_ylim(0.05, 30); axs[0].set_xlabel("lectures"); axs[0].set_ylabel("multiplicateur du pas")
axs[0].set_title("MeZO : pas réel = η · g(t)\n(gain de la graine 0, reconstitué)", loc="left", fontsize=10)
axs[0].text(0.03, 0.97, "g ≈ 0,26 en moyenne sur les 1 000 premières lectures :\nle pas réel est ~4 fois plus petit que prévu",
            transform=axs[0].transAxes, ha="left", va="top", fontsize=8.5, color=MUTED)
c = [x[1] for x in F2[("color_stable", ETA_F["color_stable"], 0)]["curve"]]
r = [x[0] for x in F2[("color_stable", ETA_F["color_stable"], 0)]["curve"]]
axs[1].plot(r, 1 / np.array(c), color=ORANGE, marker="o", ms=4)
axs[1].axhline(1, color=MUTED, lw=1, ls="--")
axs[1].set_yscale("log"); axs[1].set_ylim(0.1, 12); axs[1].set_xlabel("lectures"); axs[1].set_ylabel("multiplicateur du pas")
axs[1].set_title("ColorMeZO : pas réel ≈ η / ℓ(t)\n(perte de validation, graine 0, capteur stable)", loc="left", fontsize=10)
axs[1].text(0.03, 0.97, "ℓ ≈ 6 au départ : 6 fois plus lent que MeZO ;\nℓ < 1 en fin de course : plus rapide",
            transform=axs[1].transAxes, ha="left", va="top", fontsize=8.5, color=MUTED)
fig.tight_layout()
save(fig, "schema_pas_effectif.svg")


# ------------------------------------------------------------ 3. courbes SST-2
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
for cfg in METHODS:
    S = [s for s in SEEDS if (cfg, ETA_F[cfg], s) in F2]
    xs = [p[0] for p in F2[(cfg, ETA_F[cfg], S[0])]["curve"]]
    for j in (0, 1):
        Y = np.array([[p[1 + j] for p in F2[(cfg, ETA_F[cfg], s)]["curve"]] for s in S])
        m, sd = Y.mean(0), Y.std(0)
        axs[j].plot(xs, m, color=COL[cfg], ls=LS[cfg], label=LAB[cfg])
        axs[j].fill_between(xs, m - sd, m + sd, color=COL[cfg], alpha=0.10, lw=0)
axs[0].set_yscale("log"); axs[0].set_ylabel("perte de validation (CE du mot-étiquette)")
axs[1].set_ylabel("précision de validation")
for a in axs:
    a.set_xlabel("lectures du capteur")
axs[0].legend(fontsize=8.5, loc="upper right")
axs[0].set_title("Perte", loc="left"); axs[1].set_title("Précision (872 phrases)", loc="left")
fig.suptitle("SST-2, OPT-125M, η = 3·10⁻⁶ pour toutes les méthodes, 3 graines (bande : ± 1 écart-type)",
             x=0.01, ha="left", fontsize=10.5, color=MUTED)
fig.tight_layout()
save(fig, "llm_sst2_courbes.svg")


# ------------------------------------------------------------ 4. coût de la dérive, graine par graine
fig, axs = plt.subplots(2, len(SEEDS), figsize=(12, 5.6), sharex=True, sharey="row",
                        gridspec_kw={"height_ratios": [1, 1.6]})
for j, s in enumerate(SEEDS):
    lg = lng_path(s, 2000)
    axs[0, j].plot(np.arange(1, 2001), lg, color=MUTED, lw=1.5)
    axs[0, j].axhline(0, color=GRID, lw=1)
    axs[0, j].set_title(f"graine {s} · gain moyen sur 1 000 lectures : {np.exp(lg[:1000]).mean():.2f}",
                        loc="left", fontsize=9.5)
    for m, (d, st) in {"mezo": ("mezo_drift", "mezo_stable"), "color": ("color_drift", "color_stable")}.items():
        cd = F2[(d, ETA_F[d], s)]["curve"]; cs = F2[(st, ETA_F[st], s)]["curve"]
        xs = [p[0] for p in cd]
        axs[1, j].plot(xs, [a[1] - b[1] for a, b in zip(cd, cs)], color=COL[d], marker="o", ms=4,
                       label="MeZO" if m == "mezo" else "ColorMeZO")
    axs[1, j].axhline(0, color=MUTED, lw=1)
    axs[1, j].set_xlabel("lectures")
axs[0, 0].set_ylabel("ln g(t)")
axs[1, 0].set_ylabel("coût de la dérive\n(perte dérivant − perte stable)")
axs[1, 0].legend(fontsize=9)
fig.suptitle("Même graine, même η, même trajectoire de gain : ce que la dérive coûte à chaque méthode",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout()
save(fig, "llm_cout_derive.svg")


# ------------------------------------------------------------ 5. réglage de η
# 5a. campagne 1 (écrêtée) : rien ne bouge
fig, ax = plt.subplots(figsize=(6, 3.8))
etas1 = sorted({e for (_, e, _) in T1}, key=float)
for cfg in METHODS:
    y = [np.mean([T1[(cfg, e, s)]["final_loss"] for s in (101, 102)]) for e in etas1]
    ax.plot([float(e) for e in etas1], y, color=COL[cfg], ls=LS[cfg], marker="o", ms=4, label=LAB[cfg])
ax.set_xscale("log"); ax.set_xlabel("η"); ax.set_ylim(0, 6.5)
ax.set_ylabel("perte de réglage après 1 000 lectures")
ax.set_title(f"Campagne 1, pas bornés en norme : {np.mean([v['clipped_frac'] for v in T1.values()]):.0%} "
             "des pas écrêtés", loc="left", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
save(fig, "llm_campagne1.svg")

# 5b. grille complète (9 η) : perte et précision de réglage
etas3 = sorted({e for (_, e, _) in T3}, key=float)
X = [float(e) for e in etas3]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
for cfg in METHODS:
    L, A = [], []
    for e in etas3:
        l = [T3[(cfg, e, s)]["final_loss"] for s in (101, 102)]
        L.append(np.mean(l) if all(np.isfinite(l)) else np.nan)
        A.append(np.mean([T3[(cfg, e, s)]["curve"][-1][2] for s in (101, 102)]))
    axs[0].plot(X, L, color=COL[cfg], ls=LS[cfg], marker="o", ms=4, label=LAB[cfg])
    axs[1].plot(X, A, color=COL[cfg], ls=LS[cfg], marker="o", ms=4)
    if cfg in ("mezo_stable", "color_stable", "mezo_drift", "color_drift"):
        i = int(np.nanargmax(A))
        axs[1].plot(X[i], A[i], "o", ms=11, mfc="none", mec=COL[cfg], mew=2)
    if np.isnan(L[-1]):
        axs[0].plot(X[-1], 6.3, "x", color=COL[cfg], ms=10, mew=2.5)
        axs[0].text(X[-1], 5.75, "diverge", ha="center", fontsize=8.5, color=MUTED)
axs[0].axhline(np.log(2), color=MUTED, lw=1, ls="--")
axs[0].text(1.2e-8, np.log(2) - 0.42, "ln 2 : au-dessus, le modèle peut ignorer la phrase", fontsize=8.5, color=MUTED)
axs[0].set_ylim(0, 6.6); axs[0].set_ylabel("perte de réglage après 1 000 lectures")
axs[0].set_title("Perte : le critère de réglage du carnet", loc="left", fontsize=10)
axs[1].axhline(0.508, color=MUTED, lw=1, ls="--")
axs[1].text(1.2e-8, 0.52, "hasard (classe majoritaire)", fontsize=8.5, color=MUTED)
axs[1].set_ylabel("précision de réglage après 1 000 lectures")
axs[1].set_title("Précision (cercles : meilleur η de chaque méthode)", loc="left", fontsize=10)
for a in axs:
    a.set_xscale("log"); a.set_xlabel("η")
axs[0].legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.9))
fig.suptitle("Réglage sur 9 valeurs de η, 2 graines de réglage", x=0.01, ha="left", fontsize=10.5, color=MUTED)
fig.tight_layout()
save(fig, "llm_reglage_eta.svg")

# 5c. les η choisis par la perte : plateau sans polarité, et divergence de MeZO
fig, axs = plt.subplots(1, 2, figsize=(12, 4.3))
for s, mk in zip(SEEDS, ("o", "s", "^")):
    for cfg, e in (("mezo_drift", "1e-05"), ("color_drift", "1e-05")):
        c = F3[(cfg, e, s)]["curve"]
        axs[0].plot([p[0] for p in c], [p[1] for p in c], color=COL[cfg], marker=mk, ms=4, lw=1.6,
                    label=f"{'MeZO' if cfg.startswith('mezo') else 'ColorMeZO'} dérivant, graine {s}")
axs[0].axhline(np.log(2), color=MUTED, lw=1, ls="--")
axs[0].set_yscale("log"); axs[0].set_xlabel("lectures"); axs[0].set_ylabel("perte de validation")
axs[0].set_title("η = 10⁻⁵, capteur dérivant, chaque graine", loc="left", fontsize=10)
axs[0].legend(fontsize=7.5, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.17))
axs[0].text(30, np.log(2) * 1.03, "ln 2", ha="left", va="bottom", fontsize=8.5, color=MUTED)
for cfg, e, lab in (("mezo_stable", "3e-06", "MeZO stable, η = 3·10⁻⁶ (référence)"),
                    ("color_stable", "3e-05", "ColorMeZO stable, η = 3·10⁻⁵"),
                    ("color_drift", "1e-05", "ColorMeZO dérivant, η = 10⁻⁵"),
                    ("mezo_drift", "1e-05", "MeZO dérivant, η = 10⁻⁵")):
    Y = np.array([[p[2] for p in F3[(cfg, e, s)]["curve"]] for s in SEEDS])
    xs = [p[0] for p in F3[(cfg, e, SEEDS[0])]["curve"]]
    axs[1].plot(xs, Y.mean(0), color=COL[cfg], ls="--" if cfg == "mezo_stable" else LS[cfg], label=lab)
axs[1].axhline(0.508, color=MUTED, lw=1, ls=":")
axs[1].set_xlabel("lectures"); axs[1].set_ylabel("précision de validation (moyenne de 3 graines)")
axs[1].set_title("Précision aux η choisis par la perte", loc="left", fontsize=10)
axs[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.tight_layout()
save(fig, "llm_eta_grand.svg")


# ------------------------------------------------------------ 6. MeZO sous dérive = MeZO stable à η·ḡ
etas = sorted({e for (_, e, _) in T2}, key=float)
E = np.log10([float(e) for e in etas])
obs, pred, lab = [], [], []
for s in (101, 102):
    gbar = np.exp(lng_path(s, 1000)).mean()
    st = [T2[("mezo_stable", e, s)]["final_loss"] for e in etas]
    for i, e in enumerate(etas):
        x = E[i] + np.log10(gbar)
        if E[0] <= x <= E[-1]:
            obs.append(T2[("mezo_drift", e, s)]["final_loss"]); pred.append(float(np.interp(x, E, st)))
            lab.append(s)
obs, pred, lab = map(np.array, (obs, pred, lab))
fig, ax = plt.subplots(figsize=(5.2, 4.6))
for s, mk in ((101, "o"), (102, "s")):
    g = np.exp(lng_path(s, 1000)).mean()
    ax.plot(pred[lab == s], obs[lab == s], mk, color=BLUE, ms=8, ls="none", mec="white", mew=1.5,
            label=f"graine de réglage {s} (ḡ = {g:.2f})")
lim = [0.5, 6.5]
ax.plot(lim, lim, color=MUTED, lw=1, ls="--")
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("prédit : MeZO stable avec η · ḡ"); ax.set_ylabel("observé : MeZO sous dérive")
ax.set_title("Le coût de la dérive pour MeZO\n= un pas multiplié par le gain moyen", loc="left", fontsize=10)
ax.legend(fontsize=8.5, loc="upper left")
save(fig, "llm_gain_effectif.svg")
print(f"pas effectif : écart médian |observé − prédit| = {np.median(np.abs(obs - pred)):.3f} "
      f"sur {len(obs)} points (max {np.abs(obs - pred).max():.3f})")


# ------------------------------------------------------------ 7. finale à η réglé, 5 graines (campagne 4)
F4 = runs(load("campagne4_finale_5graines.json"), "final")
ETA4 = {cfg: e for (cfg, e, _) in F4}
SEEDS4 = sorted({s for (_, _, s) in F4})
for (cfg, e, s), v in F4.items():
    if cfg.endswith("drift"):
        assert abs(np.abs(lng_path(s, v["reads"])).max() - v["ln_g_max"]) < 1e-9, (cfg, s)

fig, axs = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
for cfg in METHODS:
    xs = [p[0] for p in F4[(cfg, ETA4[cfg], SEEDS4[0])]["curve"]]
    Y = np.array([[p[2] for p in F4[(cfg, ETA4[cfg], s)]["curve"]] for s in SEEDS4])
    m, sd = Y.mean(0), Y.std(0)
    eta_txt = {"1e-06": "10⁻⁶", "3e-06": "3·10⁻⁶"}[ETA4[cfg]]
    extra = ", confondu avec MeZO stable" if cfg == "oracle" else ""
    axs[0].plot(xs, m, color=COL[cfg], ls=LS[cfg], label=f"{LAB[cfg]} (η = {eta_txt}{extra})")
    axs[0].fill_between(xs, m - sd, m + sd, color=COL[cfg], alpha=0.10, lw=0)
axs[0].axhline(0.5, color=GRID, lw=1)
axs[0].set_xlabel("lectures du capteur"); axs[0].set_ylabel("précision de validation (872 phrases)")
axs[0].legend(fontsize=8.3, loc="upper left")
axs[0].set_title("Précision, moyenne sur 5 graines (bande : ± 1 écart-type)", loc="left", fontsize=10)

x = np.arange(len(SEEDS4))
for k, (d, st, lab, dx) in enumerate((("mezo_drift", "mezo_stable", "MeZO", -0.18),
                                       ("color_drift", "color_stable", "ColorMeZO", 0.18))):
    cost = []
    for s in SEEDS4:
        a = dict((p[0], p[2]) for p in F4[(d, ETA4[d], s)]["curve"])[2000]
        b = dict((p[0], p[2]) for p in F4[(st, ETA4[st], s)]["curve"])[2000]
        cost.append(a - b)
    axs[1].bar(x + dx, cost, width=0.34, color=COL[d], label=lab)
axs[1].axhline(0, color=MUTED, lw=1)
axs[1].set_axisbelow(True)
axs[1].set_xticks(x, [f"graine {s}\nḡ = {np.exp(lng_path(s, 2000)).mean():.2f}" for s in SEEDS4], fontsize=8.5)
axs[1].set_ylabel("coût de la dérive en précision\n(dérivant − stable, à 2 000 lectures)")
axs[1].legend(fontsize=9, loc="lower left")
axs[1].set_title("Ce que la dérive coûte, graine par graine", loc="left", fontsize=10)
fig.suptitle("SST-2, OPT-125M, η réglé par méthode sur la précision de réglage : MeZO 10⁻⁶, ColorMeZO 3·10⁻⁶",
             x=0.01, ha="left", fontsize=10.5, color=MUTED)
fig.tight_layout()
save(fig, "llm_finale_5graines.svg")


# ------------------------------------------------------------ 8. banc phase sur le jouet (partie 7)
with open(os.path.join(ROOT, "results", "bench_phase_50graines.json")) as f:
    PH = json.load(f)["results"]
rows = [("mezo_oracle", "MeZO oracle", INK), ("mezo_drift", "MeZO, capteur dérivant", BLUE),
        ("color_drift", "ColorMeZO, dérivant", ORANGE), ("color_stable", "ColorMeZO, stable", ORANGE),
        ("phase_rps2_drift", "phase, 2 lectures/pas, dérivant", "#7a5bb5"),
        ("phase_rps2_stable", "phase, 2 lectures/pas, stable", "#7a5bb5"),
        ("phase_rps1_drift", "phase, 1 lecture/pas, dérivant", "#7a5bb5")]


def wilson(k, n, z=1.96):
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


fig, ax = plt.subplots(figsize=(8.2, 3.9))
ax.set_axisbelow(True)
for i, (key, lab, col) in enumerate(rows):
    r = PH[key]; k, n = r["n_success"], r["n"]
    lo, hi = wilson(k, n)
    y = len(rows) - 1 - i
    ax.barh(y, k / n, color=col, alpha=0.35 if key.endswith("stable") else 0.9, height=0.62)
    ax.plot([lo, hi], [y, y], color=MUTED, lw=1.4)
    ax.text(hi + 0.01, y, f"{k}/{n}", va="center", fontsize=9, color=INK)
ax.set_yticks(range(len(rows)), [lab for _, lab, _ in rows][::-1], fontsize=9)
ax.set_xlim(0, 0.8); ax.set_xlabel("taux de succès (barre : intervalle de Wilson à 95 %)")
ax.grid(axis="y", visible=False)
ax.set_title("Jouet, σ_g = 0,05 par lecture, 6 000 lectures, 50 graines", loc="left", fontsize=10)
fig.tight_layout()
save(fig, "phase_jouet.svg")
