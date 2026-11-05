"""Passe avant OPT-125M en NumPy + petit test MeZO à lectures couleur."""
import time
import numpy as np
from opt_numpy import load_opt, BPETokenizer

SD = None
TOK = None


def init():
    global SD, TOK
    if SD is None:
        import gc
        SD = load_opt()   # déjà fp32 (conversion faite au rebuild)
        gc.collect()
        TOK = BPETokenizer()
    return SD, TOK


def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def attention(x, p, n_heads=12):
    B, T, D = x.shape
    hd = D // n_heads
    q = x @ p["q_proj.weight"].T + p["q_proj.bias"]
    k = x @ p["k_proj.weight"].T + p["k_proj.bias"]
    v = x @ p["v_proj.weight"].T + p["v_proj.bias"]
    q = q.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    k = k.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    v = v.reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd)
    mask = np.triu(np.full((T, T), -1e9), 1)
    att = softmax(att + mask)
    o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return o @ p["out_proj.weight"].T + p["out_proj.bias"]


def forward(input_ids):
    """input_ids : (B, T) int. Renvoie logits (B, T, V) et cache des poids pour MeZO."""
    sd = SD
    B, T = input_ids.shape
    h = sd["model.decoder.embed_tokens.weight"][input_ids] + \
        sd["model.decoder.embed_positions.weight"][2:2 + T]
    for i in range(12):
        p = {k.split(".")[-2] + "." + k.split(".")[-1]: v
             for k, v in sd.items() if k.startswith(f"model.decoder.layers.{i}.")}
        sa = p.pop("self_attn_layer_norm.weight"); sab = p.pop("self_attn_layer_norm.bias")
        h = h + attention(layernorm(h, sa, sab), p)
        ln = layernorm(h, p["final_layer_norm.weight"], p["final_layer_norm.bias"])
        h = h + (np.maximum(ln @ p["fc1.weight"].T + p["fc1.bias"], 0)
                 @ p["fc2.weight"].T + p["fc2.bias"])
    h = layernorm(h, sd["model.decoder.final_layer_norm.weight"],
                  sd["model.decoder.final_layer_norm.bias"])
    return h @ sd["lm_head.weight"].T


def param_vec():
    """Vecteur plat de tous les poids (pour la sonde SPSA)."""
    return [v for k, v in SD.items() if "embed_positions" not in k]


def n_params():
    return sum(v.size for v in param_vec())


if __name__ == "__main__":
    sd, tok = init()
    prompt = "The capital of France is"
    ids = np.array([tok.encode(prompt)])
    t0 = time.time()
    logits = forward(ids)
    dt = time.time() - t0
    next_id = int(logits[0, -1].argmax())
    # décodage glouton rapide du token
    inv = {v: k for k, v in tok.encoder.items()}
    u2b = tok.u2b
    word = "".join(inv.get(next_id, "?") or "")
    word = bytes(u2b[c] for c in word if c in u2b).decode("utf-8", "replace")
    print(f"passe avant : {dt:.1f}s | '{prompt}' -> '{word.strip()}'")
