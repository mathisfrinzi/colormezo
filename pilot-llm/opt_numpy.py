"""OPT-125M en NumPy pur : chargeur pytorch_model.bin + tokenizer GPT-2 BPE."""
import io, json, pickle, zipfile
import numpy as np


class _TorchUnpickler(pickle.Unpickler):
    """Dé-sérialise le pickle torch en interceptant storage et _rebuild_tensor_v2."""

    def __init__(self, f, zf):
        super().__init__(f)
        self.zf = zf
        self.storages = {}

    def persistent_load(self, pid):
        # pid = ('storage', storage_type, key, location, _)
        _, stype, key, _loc, _ = pid
        dtype = self._dtype(stype if isinstance(stype, bytes) else stype.__name__)
        with self.zf.open(f"archive/data/{key}") as fh:
            arr = np.frombuffer(fh.read(), dtype=dtype)
        self.storages[key] = arr
        return key

    def find_class(self, module, name):
        if name == "_rebuild_tensor_v2":
            def rebuild(storage, offset, size, stride, *_a):
                arr = self.storages[storage]
                # conversion fp32 immédiate : le buffer fp16 est libéré de suite
                # (limite cgroup 1 Gio, le pic doit rester bas)
                out = arr[offset:offset + int(np.prod(size))].reshape(size)
                if out.dtype == np.float16:
                    out = out.astype(np.float32)
                self.storages[storage] = None  # libère le buffer fp16
                return out
            return rebuild
        # type muet identifiable par son nom (HalfStorage, FloatStorage, ...)
        return type(name, (), {"__module__": module})

    def _dtype(self, stype):
        return {b"HalfStorage": np.float16, b"Half": np.float16,
                b"FloatStorage": np.float32, b"Float": np.float32,
                b"LongStorage": np.int64, b"Long": np.int64}[stype.encode() if isinstance(stype, str) else stype]


def load_opt(path="pytorch_model.bin"):
    zf = zipfile.ZipFile(path)
    with zf.open("archive/data.pkl") as fh:
        sd = _TorchUnpickler(fh, zf).load()
    return sd


# ------------------------------------------------------------- tokenizer GPT-2
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


class BPETokenizer:
    def __init__(self, vocab_path="vocab.json", merges_path="merges.txt"):
        with open(vocab_path, encoding="utf-8") as f:
            self.encoder = json.load(f)
        with open(merges_path, encoding="utf-8") as f:
            lines = f.read().split("\n")
        merges = [m for m in lines[1:] if " " in m]
        self.ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.b2u = bytes_to_unicode()
        self.u2b = {v: k for k, v in self.b2u.items()}
        self.n_vocab = len(self.encoder)

    def _bpe(self, word):
        parts = list(word)
        while len(parts) > 1:
            pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
            best = min(pairs, key=lambda p: self.ranks.get(p, 10 ** 9))
            if best not in self.ranks:
                break
            i = pairs.index(best)
            parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
        return parts

    def encode(self, text):
        import re
        ids = []
        for m in re.findall(r"'s|'t|'re|'ve|'m|'ll|'d| ?[a-zA-Z]+| ?[0-9]+| ?[^\s a-zA-Z0-9]+|\s+(?!\S)|\s+", text):
            w = "".join(self.b2u[b] for b in m.encode("utf-8"))
            ids += [self.encoder[t] for t in self._bpe(w)]
        return ids


if __name__ == "__main__":
    sd = load_opt()
    tok = BPETokenizer()
    n = sum(v.size for v in sd.values())
    print("tenseurs:", len(sd), "params:", n / 1e6, "M")
    print([k for k in list(sd)[:5]])
    ids = tok.encode("The movie was terrible. Review:")
    print("tokens:", ids)
