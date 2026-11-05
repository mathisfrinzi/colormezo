"""
Pilote idée 24 sur LLM réel : OPT-125M en NumPy pur, MeZO vs ColorMeZO
sous le même capteur couleur dérivant (u = log10(g*l), 12 bits, ln g dérive
de sigma_g=0.05 par lecture).

Pré-enregistrement (pilote, échelle fumigène, pas une étude) :
- Tâche : perte LM d'une phrase fixe ("Paris is the capital of France."),
  BOS inclus. La vraie perte est mesurée à chaque pas par une passe avant
  de plus (coûteuse ici, ~2 s ; c'est la mesure, pas l'entraînement).
- Méthodes : SPSA (MeZO) sur valeurs reconstruites linéaires vs ColorMeZO
  sur différences de lectures log dépliées ; conditions : dérive par lecture
  sigma_g=0.05 (le régime de l'idée 24) et contrôle stable.
- Budget : 15 pas x 2 lectures de capteur = 30 lectures/variante,
  eps=1e-3, eta=1e-6 NON réglé (pilote : une seule valeur a priori),
  pas non borné en norme (consigné).
- Question : "le système tourne-t-il, et la dérive brouille-t-elle MeZO
  ici aussi ?", sans prétention statistique, 1 graine.

Implémentation : perturbation en place régénérée depuis la même graine ;
restauration fusionnée avec la mise à jour (depuis th-eps*z, on applique
(eps - eta*g)*z : un seul balayage de bruit par pas économisé).
Usage : python3 opt_mezo_pilot.py <cond>   avec cond dans
{lin_drift, log_drift, lin_stable, log_stable}
"""
import json
import time
import numpy as np
import opt_forward as F

SENT = "Paris is the capital of France."
EPS = 1e-3
ETA = 1e-6
SIGMA_G = 0.05
BITS = 12
STEPS = 15
CHUNK = 4_000_000

_ids_cache = None


def trainable():
    return [(k, v) for k, v in F.SD.items() if "embed_positions" not in k]


def z_blocks(v, step_seed):
    """Générateur reproductif de z par blocs (RAM limitée : pas de z entier)."""
    rng = np.random.default_rng(step_seed)
    flat = v.ravel()
    for i in range(0, flat.size, CHUNK):
        yield i, rng.normal(0, 1, min(CHUNK, flat.size - i))


def add_cz(c, step_seed):
    """Applique c*z à tous les poids (c scalaire)."""
    for k, v in trainable():
        for i, z in z_blocks(v, step_seed):
            v.ravel()[i:i + z.size] += c * z.astype(np.float32)


class Sensor:
    def __init__(self, rng, drift):
        self.rng, self.drift, self.ln_g = rng, drift, 0.0

    def read(self, loss):
        if self.drift:
            self.ln_g += self.rng.normal(0, SIGMA_G)
        u = np.log10(np.exp(self.ln_g) * max(loss, 1e-300))
        return np.round(u * (2 ** BITS)) / (2 ** BITS)


def loss_of(ids):
    lg = F.forward(ids)[0]
    T = lg.shape[0] - 1
    lp = lg[:-1][np.arange(T), ids[0, 1:]]
    return float(-np.mean(lp - np.logaddexp.reduce(lg[:-1], axis=-1)))


def run(mode, drift, seed=0):
    ids = np.array([[2] + F.TOK.encode(SENT)])
    sensor = Sensor(np.random.default_rng(seed + 777), drift)
    hist = []
    ln10 = np.log(10.0)
    t0 = time.time()
    for step in range(STEPS):
        zs = 1000 + step
        add_cz(+EPS, zs)
        u_p = sensor.read(loss_of(ids))
        add_cz(-2 * EPS, zs)          # theta+eps*z -> theta-eps*z
        u_m = sensor.read(loss_of(ids))
        if mode == "lin":
            delta = 10.0 ** u_p - 10.0 ** u_m
        else:
            d = u_p - u_m
            delta = ln10 * (d - np.round(d))
        g = delta / (2 * EPS)
        # depuis theta-eps*z : appliquer (eps - eta*g)*z = restauration + pas
        add_cz(EPS - ETA * g, zs)
        lt = loss_of(ids)
        hist.append(lt)
        print(f"  step {step}: loss {lt:.4f} | g {g:.3e} | {time.time()-t0:.0f}s",
              flush=True)
    return {"mode": mode, "drift": drift, "eta": ETA, "eps": EPS,
            "sigma_g": SIGMA_G if drift else 0.0, "steps": STEPS,
            "loss_hist": [round(h, 4) for h in hist],
            "seconds": round(time.time() - t0, 1)}


if __name__ == "__main__":
    import sys
    cond = sys.argv[1]
    mode = "lin" if cond.startswith("lin") else "log"
    drift = cond.endswith("drift")
    F.init()
    print(f"== {cond} (initiale {loss_of(np.array([[2] + F.TOK.encode(SENT)])):.4f})",
          flush=True)
    r = run(mode, drift)
    with open(f"results_pilot_{cond}.json", "w") as f:
        json.dump(r, f, indent=1)
    print("saved", f"results_pilot_{cond}.json", flush=True)
