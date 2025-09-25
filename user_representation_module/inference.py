import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import pickle, os

from user_representation_module.config import get_infer_config

def _parse_args():
    p = argparse.ArgumentParser()
    # 원하면 특정 키만 덮어쓸 수 있음. (없으면 BASE로 동작)
    p.add_argument("--override", type=str, default="")
    return p.parse_args()

args = _parse_args()
config = get_infer_config(args.override)

device = config["device"]
os.makedirs(config['emb_out_dir'], exist_ok=True)

# ======== Data ========
with open(config['data_path'], 'rb') as f:
    domain_A = pickle.load(f)     # dict: user -> [item...]
with open(config['data_path1'], 'rb') as f:
    domain_B = pickle.load(f)
with open(config['data_path2'], 'rb') as f:
    domain_C = pickle.load(f)     # cross

def count_unique_items(list_of_lists):
    all_elements = set()
    for inner_list in list_of_lists:
        for element in inner_list:
            all_elements.add(element)
    return len(all_elements), max(all_elements)

_, num_item = count_unique_items(domain_C.values())

# ======== Inference Dataset ========
from user_representation_module.data import InferenceDataset
def make_loader(user_dict):
    ds = InferenceDataset(user_dict, config['max_len'], num_item)
    return DataLoader(ds, batch_size=config['batch_size'], shuffle=False,
                      num_workers=config['num_workers'], pin_memory=True,
                      collate_fn=lambda batch: (
                          [b[0] for b in batch],
                          torch.stack([b[1] for b in batch], dim=0)
                      ))

A_loader = make_loader(domain_A)
B_loader = make_loader(domain_B)
C_loader = make_loader(domain_C)

# ======== Model ========
from user_representation_module.model import BERT4Rec

EPOCH_TO_LOAD = int(config["EPOCH_TO_LOAD"])
specific_ckpt = os.path.join(config['save_path'], f"specific_encoder_ep{EPOCH_TO_LOAD}.pt")
cross_ckpt    = os.path.join(config['save_path'], f"cross_encoder_ep{EPOCH_TO_LOAD}.pt")

specific_encoder = BERT4Rec(num_item, config['hidden_units'], config['num_heads'],
                            config['num_layers'], config['max_len'], config['dropout_rate'], device).to(device)
cross_encoder = BERT4Rec(num_item, config['hidden_units'], config['num_heads'],
                         config['num_layers'], config['max_len'], config['dropout_rate'], device).to(device)

specific_encoder.load_state_dict(torch.load(specific_ckpt, map_location=device))
cross_encoder.load_state_dict(torch.load(cross_ckpt, map_location=device))

specific_encoder.eval()
cross_encoder.eval()

SPE_DOMAIN = config["SPE_DOMAIN"]
spe_loader = B_loader if SPE_DOMAIN == 'B' else A_loader

@torch.no_grad()
def export_embeddings(encoder, loader):
    emb_dict = {}
    for uids, toks in loader:
        toks = toks.to(device)
        seq_out = encoder(toks)              # (B, L, D)
        emb = seq_out.mean(dim=1)            # (B, D)  — 세션 레벨
        for uid, v in zip(uids, emb.detach().cpu().numpy()):
            emb_dict[uid] = v
    return emb_dict

# ======== 추출 & 저장 ========
spe_emb = export_embeddings(specific_encoder, spe_loader)
cross_emb = export_embeddings(cross_encoder, C_loader)

def dict_to_array(emb_dict):
    n = len(emb_dict)
    D = len(next(iter(emb_dict.values())))
    arr = np.zeros((n, D), dtype=np.float32)
    for uid, vec in emb_dict.items():
        arr[uid] = vec
    return arr

spe_arr = dict_to_array(spe_emb)
cross_arr = dict_to_array(cross_emb)

tag = f"ep{EPOCH_TO_LOAD}_{'B' if SPE_DOMAIN=='B' else 'A'}"
spe_pkl   = os.path.join(config['emb_out_dir'], f"spe_emb_{tag}.pkl")
cross_pkl = os.path.join(config['emb_out_dir'], f"cross_emb_{tag}.pkl")
spe_npy   = os.path.join(config['emb_out_dir'], f"spe_emb_{tag}.npy")
cross_npy = os.path.join(config['emb_out_dir'], f"cross_emb_{tag}.npy")

with open(spe_pkl, 'wb') as f: pickle.dump(spe_arr, f)
with open(cross_pkl, 'wb') as f: pickle.dump(cross_arr, f)
np.save(spe_npy, spe_arr)
np.save(cross_npy, cross_arr)

print(f"[✓] Saved specific embeddings: {spe_arr.shape} → {spe_pkl}, {spe_npy}")
print(f"[✓] Saved cross    embeddings: {cross_arr.shape} → {cross_pkl}, {cross_npy}")
