"""
Campagne idée 24 : le vrai banc LLM : MeZO vs ColorMeZO sur SST-2
(OPT-125M, NumPy pur, zéro PyTorch).

Pré-enregistrement (écrit avant toute mesure : le protocole du README,
« campagne complète », reste à exécuter ; aucune de ces valeurs n'a été
regardée au moment d'écrire ce qui suit) :

- Question Q3 : sur une vraie tâche de classification (SST-2, few-shot
  32 exemples, éval sur 250 dev), sous capteur à gain dérivant
  (u = log10(g*l) quantifié 12 bits, ln g dérive de sigma_g = 0,05 par
  lecture), ColorMeZO réduit-il mieux la perte et augmente-t-il mieux
  l'accuracy que MeZO lisant le même capteur ? Et à capteur stable, MeZO
  rejoint-il l'oracle ?

- Tâche : SST-2 (GLUE). Entraînement : 32 exemples tirés du train
  (16 par classe, rng fixé seed 1234). Template zéro-coup :
  "Review: {phrase}\\nSentiment:" ; verbaliseurs " positive" / " negative"
  (un seul token chacun, vérifié au lancement). Perte d'entraînement :
  softmax binaire sur les logits des deux verbaliseurs à la dernière
  position du prompt. Évaluation : accuracy sur les 250 premières phrases
  du dev (ensemble fixe, écrit ici).

- Méthodes : mezo_lin (différences de valeurs reconstruites 10^u : la
  décennie est connue, hypothèse identique au jouet), colormezo
  (différences de lectures log dépliées ± 0,5 décennie), oracle (perte
  exacte, une fois par graine, pas de capteur).

- Conditions : capteur dérivant (sigma_g = 0,05/lecture) et capteur
  stable (contrôle, comme C1/C2 du jouet).

- Réglage : grille d'eta {1e-6, 3e-6, 1e-5, 3e-5} par méthode x condition,
  runs courts (40 pas) sur la graine 0, sélection par perte vraie finale
  minimale. Puis campagne : 500 pas x 2 lectures = 1000 lectures par run
  (budget du README : 1000-2000), 3 graines (0-2), eps = 1e-3, pas non
  borné. L'oracle utilise l'eta choisi pour mezo_lin stable.

- Critère de succès par run (écrit avant) : accuracy dev finale >=
  accuracy initiale + 2 points. Comparaison principale (Q3) : par graine,
  delta-accuracy de ColorMeZO > delta-accuracy de MeZO sous capteur
  dérivant (n = 3 : descriptif, comparaison appariée, pas de test formel).

- Mesures : perte vraie d'entraînement toutes les 25 pas (passe avant
  supplémentaire ; c'est la mesure, pas l'entraînement), accuracy dev à
  l'init, à mi-course et à la fin.

- Checkpoints : un JSON par run dans campaign_results/ ; les runs déjà
  présents sont sautés (reprise sûre). Les poids sont rechargés depuis le
  disque à chaque run (limite mémoire ~1 Gio).

Usage :
    python3 campaign_sst2.py sanity          # fichiers + verbaliseurs + acc init
    python3 campaign_sst2.py tune            # réglage eta (graine 0)
    python3 campaign_sst2.py run             # 3 graines x conditions
    python3 campaign_sst2.py analyze          # tableau récapitulatif
Options : --steps 500 --tune-steps 40 --seeds 3 --eta-grid 1e-6,3e-6,1e-5,3e-5
Prérequis dans le répertoire : pytorch_model.bin, vocab.json, merges.txt,
SST-2/train.tsv, SST-2/dev.tsv, opt_numpy.py, opt_forward.py.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import opt_forward as F

SIGMA_G = 0.05
BITS = 12
EPS = 1e-3
CHUNK = 4_000_000
BATCH = 16
EVAL_BATCH = 16
N_DEV = 250
RNG_SHOTS = 1234
TEMPLATE = "Review: {s}\nSentiment:"
VERB_POS, VERB_NEG = " positive", " negative"

RESULTS = "campaign_results"


# --------------------------------------------------------------------- données
def forward_sel(ids, last_idx, cols):
    """Passe avant OPT qui ne calcule que les logits des colonnes `cols`
    (les deux verbaliseurs) à la dernière position utile de chaque
    exemple -> (B, len(cols)). Le lm_head sur toutes les positions du
    vocabulaire (B, T, 50272) ferait > 200 Mo de sortie ET un packing
    BLAS ~300 Mo : interdit ici (limite cgroup 1 Gio), et inutile pour
    une perte de classification."""
    sd = F.SD
    B, T = ids.shape
    h = sd["model.decoder.embed_tokens.weight"][ids] + \
        sd["model.decoder.embed_positions.weight"][2:2 + T]
    for i in range(12):
        p = {k.split(".")[-2] + "." + k.split(".")[-1]: v
             for k, v in sd.items() if k.startswith(f"model.decoder.layers.{i}.")}
        sa = p.pop("self_attn_layer_norm.weight"); sab = p.pop("self_attn_layer_norm.bias")
        h = h + F.attention(F.layernorm(h, sa, sab), p)
        ln = F.layernorm(h, p["final_layer_norm.weight"], p["final_layer_norm.bias"])
        h = h + (np.maximum(ln @ p["fc1.weight"].T + p["fc1.bias"], 0)
                 @ p["fc2.weight"].T + p["fc2.bias"])
    h = F.layernorm(h, sd["model.decoder.final_layer_norm.weight"],
                    sd["model.decoder.final_layer_norm.bias"])
    hl = h[np.arange(B), np.asarray(last_idx)]
    return hl @ sd["lm_head.weight"][cols].T


def load_sst2():
    def read(path):
        rows = []
        with open(path, encoding="utf-8") as f:
            r = csv.reader(f, delimiter="\t")
            next(r)
            for s, lab in r:
                rows.append((s.strip(), int(lab)))
        return rows
    train = read(os.path.join("SST-2", "train.tsv"))
    dev = read(os.path.join("SST-2", "dev.tsv"))[:N_DEV]
    rng = np.random.default_rng(RNG_SHOTS)
    shots = []
    for lab in (0, 1):
        pool = [i for i, (_, l) in enumerate(train) if l == lab]
        idx = rng.choice(pool, size=16, replace=False)
        shots += [train[i] for i in idx]
    # dev encodé une fois (paires (ids, label)) ; shots restent bruts
    # (train_loss les encode à chaque lecture)
    dev_enc = [(encode_prompt(F.TOK, s), lab) for s, lab in dev]
    return shots, dev_enc


def encode_prompt(tok, s):
    return [2] + tok.encode(TEMPLATE.format(s=s))


def batches(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def make_batch(encoded):
    """Right-padding + index de dernière position (le masque causal rend
    le padding de droite inoffensif pour les positions réelles)."""
    ids = [p[0] for p in encoded]
    labs = [p[1] for p in encoded]
    T = max(len(x) for x in ids)
    arr = np.zeros((len(ids), T), dtype=np.int64)
    last = []
    for i, x in enumerate(ids):
        arr[i, :len(x)] = x
        last.append(len(x) - 1)
    return arr, last, labs


def loss_acc(sel, lab):
    """sel : (B, 2) logits verbaliseurs ; perte softmax binaire + accuracy."""
    lse = np.logaddexp.reduce(sel, axis=1)
    loss = float(-np.mean(sel[np.arange(len(lab)), np.asarray(lab)] - lse))
    acc = float(np.mean((sel[:, 1] > sel[:, 0]) == np.asarray(lab, dtype=bool)))
    return loss, acc


def eval_batches(pairs, verb_ids):
    """Perte/accuracy moyennes sur un ensemble encodé."""
    losses, accs, n = [], [], 0
    for bat in batches(pairs, EVAL_BATCH):
        arr, last, labs = make_batch(bat)
        l, a = loss_acc(forward_sel(arr, last, verb_ids), labs)
        losses.append(l * len(bat))
        accs.append(a * len(bat))
        n += len(bat)
    return sum(losses) / n, sum(accs) / n


def train_loss(shots, verb_ids):
    """Perte d'entraînement vraie sur les 32 few-shots (2 batches de 16)."""
    pairs = [(encode_prompt(F.TOK, s), lab) for s, lab in shots]
    losses = []
    for bat in batches(pairs, BATCH):
        arr, last, labs = make_batch(bat)
        l, _ = loss_acc(forward_sel(arr, last, verb_ids), labs)
        losses.append(l)
    return float(np.mean(losses))


# ---------------------------------------------------------------- sonde SPSA
def trainable():
    return [(k, v) for k, v in F.SD.items() if "embed_positions" not in k]


def z_blocks(v, step_seed):
    rng = np.random.default_rng(step_seed)
    flat = v.ravel()
    for i in range(0, flat.size, CHUNK):
        yield i, rng.normal(0, 1, min(CHUNK, flat.size - i))


def add_cz(c, step_seed):
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


def reload_model():
    import gc
    F.SD = None
    F.init()
    gc.collect()


# ------------------------------------------------------------------- un run
def run(method, drift, seed, eta, steps, shots, dev, verb_ids, tag, verbose=True):
    """method : lin | log | oracle ; drift : bool ; renvoie le dict de run."""
    reload_model()
    ln10 = np.log(10.0)
    sensor = Sensor(np.random.default_rng(seed + 777), drift and method != "oracle")
    l0 = train_loss(shots, verb_ids)
    loss0, acc0 = eval_batches(dev, verb_ids)
    hist = [l0]
    t0 = time.time()
    for step in range(steps):
        zs = 10000 + 100000 * seed + step
        add_cz(+EPS, zs)
        l_p_true = train_loss(shots, verb_ids)
        u_p = sensor.read(l_p_true)
        add_cz(-2 * EPS, zs)
        l_m_true = train_loss(shots, verb_ids)
        u_m = sensor.read(l_m_true)
        if method == "lin":
            delta = 10.0 ** u_p - 10.0 ** u_m
        elif method == "log":
            d = u_p - u_m
            delta = ln10 * (d - np.round(d))
        else:  # oracle : différence linéaire de la perte exacte
            delta = l_p_true - l_m_true
        g = delta / (2 * EPS)
        add_cz(EPS - eta * g, zs)
        if (step + 1) % 25 == 0 or step == steps - 1:
            lt = train_loss(shots, verb_ids)
            hist.append(lt)
            if verbose and (step + 1) % 100 == 0:
                print(f"    [{tag}] pas {step+1}/{steps} : perte {lt:.4f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    lossf, accf = eval_batches(dev, verb_ids)
    out = {"tag": tag, "method": method, "drift": bool(drift and method != "oracle"),
           "seed": seed, "eta": eta, "eps": EPS, "steps": steps,
           "train_loss_init": round(l0, 4), "train_loss_final": round(hist[-1], 4),
           "train_loss_ratio": round(hist[-1] / l0, 4),
           "dev_acc_init": round(acc0, 4), "dev_acc_final": round(accf, 4),
           "d_acc_pts": round(100 * (accf - acc0), 2),
           "success": bool(accf >= acc0 + 0.02),
           "train_loss_hist": [round(h, 4) for h in hist],
           "seconds": round(time.time() - t0, 1)}
    if verbose:
        print(f"  [fini] {tag} : perte {l0:.3f} -> {hist[-1]:.3f} | "
              f"acc {100*acc0:.1f} -> {100*accf:.1f} ({out['d_acc_pts']:+.1f} pts, "
              f"{out['success']}) | {out['seconds']}s", flush=True)
    return out


def save(r, name):
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(r, f, indent=1)


def load_all(prefix):
    if not os.path.isdir(RESULTS):
        return []
    out = []
    for fn in sorted(os.listdir(RESULTS)):
        if fn.startswith(prefix) and fn.endswith(".json"):
            with open(os.path.join(RESULTS, fn)) as f:
                out.append(json.load(f))
    return out


def verb_ids():
    pos = F.TOK.encode(VERB_POS)
    neg = F.TOK.encode(VERB_NEG)
    assert len(pos) == 1 and len(neg) == 1, (pos, neg)
    return [neg[0], pos[0]]  # indice 0 = négatif, 1 = positif


# -------------------------------------------------------------------- phases
def phase_sanity(args):
    for p in ["pytorch_model.bin", "vocab.json", "merges.txt",
              os.path.join("SST-2", "train.tsv"), os.path.join("SST-2", "dev.tsv"),
              "opt_numpy.py", "opt_forward.py"]:
        if not os.path.exists(p):
            raise SystemExit(f"manquant : {p} (voir run_campaign.ps1)")
    reload_model()
    vi = verb_ids()
    shots, dev = load_sst2()
    print(f"verbaliseurs : neg={vi[0]}, pos={vi[1]}")
    print(f"few-shots : {len(shots)} | dev : {len(dev)}")
    l, a = eval_batches(dev, verb_ids())
    print(f"accuracy dev initiale : {100*a:.1f} % | perte dev : {l:.4f}")
    print("SANITY OK")


def eta_for(method, drift):
    for t in load_all("tune_choice_"):
        if t["method"] == method and t["drift"] == drift:
            return t["chosen_eta"]
    raise SystemExit(f"eta non réglé pour {method}/drift={drift} : lance `tune` d'abord")


def phase_tune(args):
    reload_model()
    vi = verb_ids()
    shots, dev = load_sst2()
    grid = [float(x) for x in args.eta_grid.split(",")]
    for method in ("lin", "log"):
        for drift in (True, False):
            best, best_eta = None, None
            for eta in grid:
                name = f"tune_{method}_{'drift' if drift else 'stable'}_{eta:g}.json"
                r = run(method, drift, 0, eta, args.tune_steps, shots, dev, vi, name)
                r["chosen"] = False
                if best is None or r["train_loss_final"] < best:
                    best, best_eta = r["train_loss_final"], eta
                save(r, name)
            print(f"[réglage] {method}/{'drift' if drift else 'stable'} : "
                  f"eta = {best_eta:g} (perte {best:.4f})", flush=True)
            save({"method": method, "drift": drift, "chosen_eta": best_eta,
                  "best_final_loss": best}, f"tune_choice_{method}_"
                  f"{'drift' if drift else 'stable'}.json")


def phase_run(args):
    reload_model()
    vi = verb_ids()
    shots, dev = load_sst2()
    for seed in range(args.seeds):
        for method, drift in (("lin", True), ("log", True),
                              ("lin", False), ("log", False)):
            eta = eta_for(method, drift)
            name = f"run_{method}_{'drift' if drift else 'stable'}_s{seed}.json"
            if os.path.exists(os.path.join(RESULTS, name)):
                print(f"[skip] {name}", flush=True)
                continue
            r = run(method, drift, seed, eta, args.steps, shots, dev, vi, name)
            save(r, name)
        eta_o = eta_for("lin", False)
        name = f"run_oracle_s{seed}.json"
        if not os.path.exists(os.path.join(RESULTS, name)):
            r = run("oracle", False, seed, eta_o, args.steps, shots, dev, vi, name)
            save(r, name)


def phase_analyze(args):
    runs = load_all("run_")
    if not runs:
        raise SystemExit("aucun run dans campaign_results/ : lance `run` d'abord")
    def key(r):
        return (r["method"], r.get("drift", False))
    lines = ["| condition | méthode | eta | pertes (ratio) | acc (pts) | succès |",
             "|:---|:---|:---:|:---:|:---:|:---:|"]
    med = {}
    for k in sorted({key(r) for r in runs}):
        rs = [r for r in runs if key(r) == k]
        ratio = np.median([r["train_loss_ratio"] for r in rs])
        dacc = np.median([r["d_acc_pts"] for r in rs])
        n_ok = sum(r["success"] for r in rs)
        etas = sorted({f"{r['eta']:g}" for r in rs})
        m, d = k
        cond = "dérive" if d else "stable"
        lines.append(f"| {cond} | {m} | {','.join(etas)} | {ratio:.3f} | "
                     f"{dacc:+.1f} | {n_ok}/{len(rs)} |")
        med[k] = (ratio, dacc)
    table = "\n".join(lines)
    print(table)
    paired = ""
    rs_lin = [r for r in runs if r["method"] == "lin" and r.get("drift")]
    rs_log = [r for r in runs if r["method"] == "log" and r.get("drift")]
    if rs_lin and rs_log:
        by_seed = lambda rs: {r["seed"]: r["d_acc_pts"] for r in rs}
        bl, bg = by_seed(rs_lin), by_seed(rs_log)
        common = sorted(set(bl) & set(bg))
        wins = sum(bg[s] > bl[s] for s in common)
        ties = sum(bg[s] == bl[s] for s in common)
        paired = (f"\nQ3 (dérive, apparié par graine, n={len(common)}) : "
                  f"ColorMeZO gagne {wins}, égalite {ties}, perd "
                  f"{len(common)-wins-ties}")
        print(paired)
    with open(os.path.join(RESULTS, "summary.md"), "w") as f:
        f.write(table + "\n" + paired + "\n")
    print("\nÉcrit dans campaign_results/summary.md")


def main():
    p = argparse.ArgumentParser(description="Campagne idée 24 : SST-2, OPT-125M NumPy pur")
    p.add_argument("phase", choices=["sanity", "tune", "run", "analyze"])
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--tune-steps", type=int, default=40)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--eta-grid", type=str, default="1e-6,3e-6,1e-5,3e-5")
    args = p.parse_args()
    {"sanity": phase_sanity, "tune": phase_tune,
     "run": phase_run, "analyze": phase_analyze}[args.phase](args)


if __name__ == "__main__":
    main()
