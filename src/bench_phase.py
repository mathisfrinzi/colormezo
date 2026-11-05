"""
Banc de la piste « phase » : la teinte comme information (idée 24, suite).

Question Q4 : si le capteur rend aussi la PHASE de la sortie complexe, une lecture
unique en θ + i·ε·z suffit-elle à estimer la pente, sans paire, et donc sans le
bruit de dérive entre les deux lectures d'une paire ?

Principe. La perte est prolongée analytiquement (tanh, produit, moyenne). En
θ + i·ε·z :   L = ℓ + i·ε·∂_zℓ − (ε²/2)·∂²_zℓ + ...
   arg L ≈ ε·∂_zℓ / ℓ     (même pente relative que ColorMeZO)
Un gain réel positif g (qui dérive) multiplie L : |gL| change, arg(gL) = arg L.
Il s'élimine donc de la phase sans aucune paire.

Comptabilité (reads_per_step) : sur un ordinateur, une évaluation complexe coûte
2 à 4 passes réelles, donc au moins une paire : reads_per_step = 2, aucun gain de
budget. Sur un capteur physique, une mesure complexe est une seule lecture :
reads_per_step = 1, deux fois plus de pas pour le même budget de lectures.
Usage : python3 src/bench_phase.py          run 2, celui du rapport (~3 min 20 CPU)
        python3 src/bench_phase.py --run1   run 1 exploratoire (~6 min CPU)
Résultats et lecture des prédictions : rapport/6_phase.md.

Prédictions (écrites avant de lancer) :
  P7a  Sans bruit de phase, la phase donne le même succès sous gain dérivant et
       stable (identité exacte : arg(gL) = arg(L)).
  P7b  Avec ε = 1e-2 et 12 bits de phase sur 2π, le pas de quantification
       (1,5e-3 rad) est du même ordre que le signal ε·pente : la phase est
       moins bonne que ColorMeZO. Avec ε = 1e-1 (biais en ε² encore faible),
       elle égale ou dépasse ColorMeZO.
  P7c  Un décalage de phase aléatoire du capteur (σ_ψ par lecture) détruit
       la méthode dès que σ_ψ approche ε·pente : c'est la limite physique réelle.
Critère de réfutation de l'intérêt de la phase : à budget égal, phase sans
bruit de phase <= ColorMeZO en taux de succès pour tous les ε testés.
"""

import json
from pathlib import Path

import numpy as np

import bench_colormezo as B

PHASE_BITS = 12
TWO_PI = 2 * np.pi


def complex_loss(v, X, y):
    net = B.vec_to_net(v)
    r = B.forward(net, X) - y            # analytique : pas de conjugaison
    return np.mean(r * r)


def read_phase(L, sensor_rng, ln_g, psi_sigma, bits):
    """Capteur complexe : g réel > 0 qui dérive, décalage de phase ψ aléatoire,
    phase quantifiée sur `bits` bits dans [-π, π)."""
    w = np.exp(ln_g) * L
    phi = np.angle(w)
    if psi_sigma > 0:
        phi = phi + sensor_rng.normal(0, psi_sigma)
    phi = (phi + np.pi) % TWO_PI - np.pi
    step = TWO_PI / (2 ** bits)
    return np.round(phi / step) * step


def run_phase(task, eta, seed, eps, bits=PHASE_BITS, psi_sigma=0.0, drift=True,
              budget=B.BUDGET, reads_per_step=2):
    X, y, teacher, v0 = task
    l_star = B.teacher_loss(X, y, teacher)
    rng = np.random.default_rng(seed + 555)           # mêmes z que les autres méthodes
    srng = np.random.default_rng(seed + 777)
    v = v0.copy()
    m = len(v)
    ln_g = 0.0
    evals = 0
    best = B.true_loss(v, X, y)
    ets = 0 if best <= B.TOL * l_star else None
    while evals < budget:
        z = rng.normal(0, 1, m)
        if drift:
            ln_g += srng.normal(0, B.SIGMA_G)
        L = complex_loss(v + 1j * eps * z, X, y)
        phi = read_phase(L, srng, ln_g, psi_sigma, bits)
        evals += reads_per_step                        # 2 : coût numérique ; 1 : capteur physique
        g = (phi / eps) * z                            # ≈ (∂_zℓ / ℓ) z
        step = -eta * g
        nrm = np.linalg.norm(step)
        if nrm > 0.5:
            step *= 0.5 / nrm
        v = v + step
        lt = B.true_loss(v, X, y)
        best = min(best, lt)
        if ets is None and lt <= B.TOL * l_star:
            ets = evals
    return {"success": ets is not None, "evals_to_success": ets, "final_ratio": best / l_star}


def tune_phase(**kw):
    best_eta, best_med = None, np.inf
    for eta in B.ETA_GRID:
        ratios = [run_phase(B.make_task(s), eta, s, **kw)["final_ratio"] for s in B.TUNE_SEEDS]
        med = float(np.median(ratios))
        if med < best_med:
            best_med, best_eta = med, eta
    return best_eta, best_med


def summarize(res):
    succ = [r["success"] for r in res]
    ev = [r["evals_to_success"] for r in res if r["success"]]
    return {"success": float(np.mean(succ)), "n_success": int(np.sum(succ)), "n": len(res),
            "ratio_median": float(np.median([r["final_ratio"] for r in res])),
            "evals_median": float(np.median(ev)) if ev else None}


def _report(results, grid):
    def report(name, eta, res):
        results[name] = dict(summarize(res), eta=eta)
        edge = "  (bord de grille)" if eta in (grid[0], grid[-1]) else ""
        print(f"{name:<36} η={eta:<7g} succès {results[name]['n_success']}/{results[name]['n']} "
              f"ratio méd. {results[name]['ratio_median']:.2f}{edge}", flush=True)
    return report


def _references(report):
    """Mêmes graines, même capteur que l'étude 1."""
    for name, (mode, drift) in {"mezo_oracle": ("exact", False), "mezo_drift": ("sensor", True),
                                "color_drift": ("color", True), "color_stable": ("color", False)}.items():
        eta, _ = B.tune(mode, drift=drift)
        report(name, eta, [B.run_spsa(B.make_task(s), eta, mode, s, drift=drift) for s in B.TEST_SEEDS])


def run1():
    """Run 1 (exploratoire) : grille de η de l'étude 1, 20 graines ; ε ∈ {1e-2, 1e-1},
    bruit de phase ψ tiré indépendamment à chaque lecture, 2 puis 1 lecture par pas."""
    results = {}
    report = _report(results, B.ETA_GRID)
    _references(report)
    for rps in (2, 1):
        for eps in (1e-2, 1e-1):
            for psi in (0.0, 1e-3, 1e-2):
                for drift in ((True, False) if psi == 0.0 and rps == 2 else (True,)):
                    kw = dict(eps=eps, psi_sigma=psi, drift=drift, reads_per_step=rps)
                    eta, _ = tune_phase(**kw)
                    name = f"phase_rps{rps}_eps{eps:g}_psi{psi:g}_{'drift' if drift else 'stable'}"
                    report(name, eta, [run_phase(B.make_task(s), eta, s, **kw) for s in B.TEST_SEEDS])
    return {"bits_phase": PHASE_BITS, "sigma_g": B.SIGMA_G, "budget": B.BUDGET, "eta_grid": B.ETA_GRID,
            "n_tune": len(B.TUNE_SEEDS), "n_test": len(B.TEST_SEEDS)}, results, "bench_phase.json"


def run2():
    """Run 2 (celui du rapport) : le η* du run 1 était au bord bas de la grille (0,003) ; grille
    étendue vers le bas et 50 graines de test. ε = 1e-2, sans bruit de phase."""
    B.ETA_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
    B.TEST_SEEDS = list(range(1000, 1050))
    results = {}
    report = _report(results, B.ETA_GRID)
    _references(report)
    for name, kw in {"phase_rps2_drift": dict(reads_per_step=2, drift=True),
                     "phase_rps2_stable": dict(reads_per_step=2, drift=False),
                     "phase_rps1_drift": dict(reads_per_step=1, drift=True)}.items():
        kw = dict(kw, eps=1e-2, psi_sigma=0.0)
        eta, _ = tune_phase(**kw)
        report(name, eta, [run_phase(B.make_task(s), eta, s, **kw) for s in B.TEST_SEEDS])
    return {"bits_phase": PHASE_BITS, "sigma_g": B.SIGMA_G, "budget": B.BUDGET, "eps": 1e-2,
            "eta_grid": B.ETA_GRID, "n_tune": len(B.TUNE_SEEDS), "n_test": len(B.TEST_SEEDS)}, \
        results, "bench_phase_50graines.json"


def main():
    import sys
    config, results, fname = run1() if "--run1" in sys.argv else run2()
    path = Path(__file__).resolve().parent.parent / "results" / fname
    path.write_text(json.dumps({"config": config, "results": results}, indent=1))
    print("écrit", path)


if __name__ == "__main__":
    main()
