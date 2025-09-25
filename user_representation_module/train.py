import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import pickle
import os
from collections import defaultdict

from user_representation_module.config import get_train_config

def _parse_args():
    p = argparse.ArgumentParser()
    # 원하면 특정 키만 덮어쓸 수 있음. (없으면 BASE로 동작)
    p.add_argument("--override", type=str, default="")
    return p.parse_args()

args = _parse_args()
config = get_train_config(args.override)

device = config["device"]
os.makedirs(config['save_path'], exist_ok=True)

# === Data Loading ===
with open(config['data_path'], 'rb') as f:
    domain_A_train = pickle.load(f)
with open(config['data_path1'], 'rb') as f:
    domain_B_train = pickle.load(f)
with open(config['data_path2'], 'rb') as f:
    domain_C_train = pickle.load(f)

def count_unique_items(list_of_lists):
    all_elements = set()
    for inner_list in list_of_lists:
        for element in inner_list:
            all_elements.add(element)
    return len(all_elements), max(all_elements)

_, num_item = count_unique_items(domain_C_train.values())

from user_representation_module.data import BERTRecDataSet
from user_representation_module.model import (
    BERT4Rec, GradReverse, DomainDiscriminator,
    masked_loss, domain_loss, alignment_loss, separation_loss
)

# === Initialize  ===
specific_encoder = BERT4Rec(num_item, config['hidden_units'], config['num_heads'], config['num_layers'], config['max_len'], config['dropout_rate'], device).to(device)
common_encoder   = BERT4Rec(num_item, config['hidden_units'], config['num_heads'], config['num_layers'], config['max_len'], config['dropout_rate'], device).to(device)
cross_encoder    = BERT4Rec(num_item, config['hidden_units'], config['num_heads'], config['num_layers'], config['max_len'], config['dropout_rate'], device).to(device)
prediction_head  = nn.Linear(config['hidden_units'], num_item + 2).to(device)
discriminator    = DomainDiscriminator(config['hidden_units']).to(device)

params = list(specific_encoder.parameters()) + list(common_encoder.parameters()) + list(cross_encoder.parameters()) + \
         list(discriminator.parameters()) + list(prediction_head.parameters())
optimizer = torch.optim.Adam(params, lr=config['lr'])

A_loader = DataLoader(BERTRecDataSet(domain_A_train, config['max_len'], len(domain_A_train), num_item, config['mask_prob']), batch_size=config['batch_size'], shuffle=True)
B_loader = DataLoader(BERTRecDataSet(domain_B_train, config['max_len'], len(domain_B_train), num_item, config['mask_prob']), batch_size=config['batch_size'], shuffle=True)
C_loader = DataLoader(BERTRecDataSet(domain_C_train, config['max_len'], len(domain_C_train), num_item, config['mask_prob']), batch_size=config['batch_size'], shuffle=True)

# === Training  ===
for epoch in range(1, config['num_epochs'] + 1):
    for (xA, yA), (xB, yB), (xC, yC) in zip(A_loader, B_loader, C_loader):
        xA, yA = xA.to(device), yA.to(device)
        xB, yB = xB.to(device), yB.to(device)
        xC, yC = xC.to(device), yC.to(device)

        z_spe_A = specific_encoder(xA)
        z_spe_B = specific_encoder(xB)
        z_com_A = common_encoder(xA)
        z_com_B = common_encoder(xB)
        z_cross = cross_encoder(xC)

        dom_label_A = torch.zeros(xA.size(0), dtype=torch.long).to(device)
        dom_label_B = torch.ones(xB.size(0), dtype=torch.long).to(device)

        loss_spe_masked = masked_loss(z_spe_A, yA, prediction_head) + masked_loss(z_spe_B, yB, prediction_head)
        loss_disc_spe = domain_loss(discriminator(z_spe_A.mean(dim=1), grl=False), dom_label_A) + \
                        domain_loss(discriminator(z_spe_B.mean(dim=1), grl=False), dom_label_B)
        loss_spe_total = loss_spe_masked + loss_disc_spe

        loss_disc_com = domain_loss(discriminator(z_com_A.mean(dim=1), grl=True), dom_label_A) + \
                        domain_loss(discriminator(z_com_B.mean(dim=1), grl=True), dom_label_B)

        loss_cross_masked = masked_loss(z_cross, yC, prediction_head)
        z_com_mean = (z_com_A.mean(dim=1).detach() + z_com_B.mean(dim=1).detach()) / 2
        z_spe_mean = (z_spe_A.mean(dim=1) + z_spe_B.mean(dim=1)) / 2
        loss_align = alignment_loss(z_cross.mean(dim=1), z_com_mean)
        loss_separate = separation_loss(z_cross.mean(dim=1), z_spe_mean)
        loss_cross_total = loss_cross_masked + loss_align + loss_separate

        optimizer.zero_grad()
        loss_spe_total.backward(retain_graph=True)
        loss_disc_com.backward(retain_graph=True)
        loss_cross_total.backward()
        optimizer.step()

    print(f"[Epoch {epoch}] spe={loss_spe_total.item():.4f}, com={loss_disc_com.item():.4f}, cross={loss_cross_total.item():.4f}")

    if epoch % 10 == 0:
        os.makedirs(config['save_path'], exist_ok=True)
        torch.save(specific_encoder.state_dict(),  os.path.join(config['save_path'], f"specific_encoder_ep{epoch}.pt"))
        torch.save(common_encoder.state_dict(),    os.path.join(config['save_path'], f"common_encoder_ep{epoch}.pt"))
        torch.save(cross_encoder.state_dict(),     os.path.join(config['save_path'], f"cross_encoder_ep{epoch}.pt"))
        torch.save(prediction_head.state_dict(),   os.path.join(config['save_path'], f"prediction_head_ep{epoch}.pt"))
        torch.save(discriminator.state_dict(),     os.path.join(config['save_path'], f"discriminator_ep{epoch}.pt"))
        print(f"[✓] Models saved for epoch {epoch} at {config['save_path']}")
