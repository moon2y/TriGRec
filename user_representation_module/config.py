import copy
from typing import Optional, Dict, Any

# train.py 기본값
TRAIN: Dict[str, Any] = {
    "data_path": "data/food_train.pkl",     # domain A
    "data_path1": "data/kitchen_all.pkl",   # domain B (target)
    "data_path2": "data/cross_f_train.pkl", # cross
    "max_len": 150,
    "hidden_units": 50,
    "num_heads": 1,
    "num_layers": 2,
    "dropout_rate": 0.3,
    "lr": 0.001,
    "batch_size": 32,
    "num_epochs": 100,
    "num_workers": 2,
    "mask_prob": 0.15,
    "save_path": "state_dict/rep/food",
    "device": "cuda:1",
}

# inference.py 기본값
INFER: Dict[str, Any] = {
    "data_path": "data/food_train.pkl",     # domain A
    "data_path1": "data/kitchen_all.pkl",   # domain B (target)
    "data_path2": "data/cross_f_train.pkl", # cross
    "max_len": 150,
    "hidden_units": 50,
    "num_heads": 1,
    "num_layers": 2,
    "dropout_rate": 0.3,
    "batch_size": 256,
    "num_workers": 2,
    "save_path": "state_dict/rep/food",     # 모델 저장된 폴더
    "emb_out_dir": "user_representation_module/embeddings/food",
    "device": "cuda:1",
    "EPOCH_TO_LOAD": 100,
    "SPE_DOMAIN": "A",  # 'B'면 kitchen, 'A'면 food
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

def get_train_config(override: Optional[str] = None) -> Dict[str, Any]:
    if not override:
        return copy.deepcopy(TRAIN)
    return _deep_update(TRAIN, _parse_override(override))

def get_infer_config(override: Optional[str] = None) -> Dict[str, Any]:
    if not override:
        return copy.deepcopy(INFER)
    return _deep_update(INFER, _parse_override(override))
