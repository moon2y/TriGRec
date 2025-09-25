import copy
from typing import Optional, Dict, Any

BASE: Dict[str, Any] = {
    "data_path": "data/food_all.pkl",

    "max_len": 10,
    "hidden_units": 50,
    "num_heads": 1,
    "num_layers": 2,
    "dropout_rate": 0.3,

    "lr": 0.001,
    "batch_size": 2048,
    "num_epochs": 100,
    "num_workers": 2,
    "mask_prob": 0.15,

    "spe_emb_path": "user_representation_module/embeddings/food/spe_emb_ep100_A.pkl",
    "cross_emb_path": "user_representation_module/embeddings/food/cross_emb_ep100_A.pkl",

    "device": "cuda:1",
    "seed": 1225,
}

def _deep_update(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    r = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(r.get(k), dict):
            r[k] = _deep_update(r[k], v)
        else:
            r[k] = v
    return r

def _parse_override(ovr: Optional[str]) -> Dict[str, Any]:
    if not ovr:
        return {}
    out: Dict[str, Any] = {}
    for pair in ovr.split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, v = pair.split("=", 1)
        cur = out
        ks = k.split(".")
        for p in ks[:-1]:
            cur = cur.setdefault(p, {})
        vv = v.strip()
        if vv.lower() in {"true", "false"}:
            val: Any = vv.lower() == "true"
        else:
            try:
                val = float(vv) if "." in vv else int(vv)
            except ValueError:
                val = vv
        cur[ks[-1]] = val
    return out

def get_config(override: Optional[str] = None) -> Dict[str, Any]:
    """Return config dict composed from BASE (+ optional --override)."""
    if not override:
        return copy.deepcopy(BASE)
    return _deep_update(BASE, _parse_override(override))
