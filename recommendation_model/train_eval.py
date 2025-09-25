import math
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from box import Box
import warnings

warnings.filterwarnings(action='ignore')
torch.set_printoptions(sci_mode=True)

import argparse
from recommendation_model.config import get_config
from recommendation_model.data import MakeSequenceDataSet, BERTRecDataSet
from recommendation_model.model import (
    ScaledDotProductAttention, MultiHeadAttention, PositionwiseFeedForward,
    BERT4RecBlock, BERT4Rec, CrossAttentionGating, BERT4RecWithCrossAttention
)

def _parse_args():
    p = argparse.ArgumentParser()
    # 원하면 특정 키만 덮어쓸 수 있음. (없으면 BASE로 동작)
    p.add_argument("--override", type=str, default="")
    return p.parse_args()

args = _parse_args()
config = Box(get_config(args.override))
device = config.device


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = config.seed
set_seed(seed)

def count_unique_items(list_of_lists):
    all_elements = set()
    for inner_list in list_of_lists:
        for element in inner_list:
            all_elements.add(element)
    return min(all_elements), max(all_elements)

item_min, num_item = count_unique_items(pd.read_pickle(config.data_path).values())

make_sequence_dataset = MakeSequenceDataSet(config = dict(config))
user_train, user_label, user_valid, cross_emb, spe_emb = make_sequence_dataset.get_train_valid_data()

bert4rec_dataset = BERTRecDataSet(
    user_train = user_train,
    max_len = config.max_len,
    num_user = make_sequence_dataset.num_user,
    num_item = num_item,
    item_min = item_min,
    mask_prob = config.mask_prob,
)

data_loader = DataLoader(
    bert4rec_dataset,
    batch_size = config.batch_size,
    shuffle = False,
    pin_memory = True,
    num_workers = config.num_workers,
)

B4R = BERT4Rec(
    num_user = make_sequence_dataset.num_user,
    num_item = num_item,
    item_min = item_min,
    hidden_units = config.hidden_units,
    num_heads = config.num_heads,
    num_layers = config.num_layers,
    max_len = config.max_len,
    dropout_rate = config.dropout_rate,
    device = device,
)

model = BERT4RecWithCrossAttention(B4R, device).to(device)

# 배치 준비
batch_size = config.batch_size
label_batches, valid_batches, cross_batches, spe_batches = [], [], [], []
label = list(user_label.values())
valid = list(user_valid.values())

for i in range(0, len(label), batch_size):
    label_batches.append(label[i:i + batch_size])
    valid_batches.append(valid[i:i + batch_size])
    cross_batches.append(cross_emb[i:i + batch_size])
    spe_batches.append(spe_emb[i:i + batch_size])

def train(model, criterion, optimizer, data_loader):
    model.train()
    loss_val = 0
    for i, seq in enumerate(data_loader):
        logits = model(seq, torch.tensor(spe_batches[i],dtype=torch.float32).to(device),
                            torch.tensor(cross_batches[i],dtype=torch.float32).to(device)).to(device)
        labels = torch.tensor(label_batches[i]).to(device)
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss_val += loss.item()
        loss.backward()
        optimizer.step()
    loss_val /= len(data_loader)
    return loss_val

def evaluate(model, user_train, user_label, user_valid, max_len, bert4rec_dataset, make_sequence_dataset):
    model.eval()
    NDCG_5 = 0.0; NDCG_10 = 0.0; HIT_1 = 0.0; HIT_5 = 0.0; HIT_10 = 0.0; MRR = 0.0
    num_item_sample = 999
    users = [user for user in range(make_sequence_dataset.num_user)]
    for user in users:
        seq = (user_train[user] + [user_label[user]] + [0])[-max_len:]
        rated = [user_valid[user]]
        items = [user_valid[user]] + bert4rec_dataset.random_neg_sampling(rated_item=rated, num_item_sample=num_item_sample)
        with torch.no_grad():
            predictions = -model(np.array([seq]),
                                 torch.tensor(spe_emb[user], dtype=torch.float32).to(device).view(1, 50),
                                 torch.tensor(cross_emb[user], dtype=torch.float32).to(device).view(1, 50)).to(device).squeeze()
            predictions = predictions[items]
            rank = predictions.argsort().argsort()[0].item()
        if rank < 1:  HIT_1 += 1
        if rank < 5:  NDCG_5 += 1 / np.log2(rank + 2); HIT_5 += 1
        if rank < 10: NDCG_10 += 1 / np.log2(rank + 2); HIT_10 += 1
        MRR += 1 / (rank + 1)
    n = len(users)
    return NDCG_5/n, NDCG_10/n, HIT_1/n, HIT_5/n, HIT_10/n, MRR/n

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

loss_list = []; ndcg_5_list = []; ndcg_10_list = []
hit_1_list = []; hit_5_list = []; hit_10_list = []; mrr_list = []

for epoch in range(1, config.num_epochs + 1):
    tbar = tqdm(range(1))
    for _ in tbar:
        train_loss = train(model, criterion, optimizer, data_loader)
        ndcg_5, ndcg_10, hit_1, hit_5, hit_10, mrr = evaluate(
            model=model,
            user_train=user_train,
            user_label=user_label,
            user_valid=user_valid,
            max_len=config.max_len,
            bert4rec_dataset=bert4rec_dataset,
            make_sequence_dataset=make_sequence_dataset,
        )
        loss_list.append(train_loss)
        ndcg_5_list.append(ndcg_5); ndcg_10_list.append(ndcg_10)
        hit_1_list.append(hit_1); hit_5_list.append(hit_5); hit_10_list.append(hit_10)
        mrr_list.append(mrr)
        tbar.set_description(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}| NDCG@5: {ndcg_5:.5f}| NDCG@10: {ndcg_10:.5f}| HIT@1: {hit_1:.5f}| HIT@5: {hit_5:.5f}| HIT@10: {hit_10:.5f}| MRR: {mrr:.5f}')

# 저장 경로
os.makedirs('state_dict/rec/food', exist_ok=True)
torch.save(model.state_dict(), 'state_dict/rec/food/maxlen10.pt')
