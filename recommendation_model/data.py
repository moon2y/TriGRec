import numpy as np
import pandas as pd
import pickle
import random
import torch
from torch.utils.data import Dataset

class MakeSequenceDataSet():
    """
    SequenceData 생성 + 외부 임베딩 로드
    """
    def __init__(self, config):
        self.config = config
        self.user_train, self.user_label, self.user_valid, self.cross, self.spe = self.generate_sequence_data()
        self.num_user = len(self.user_train)

    def _load_array(self, path):
        if path.endswith(".npy"):
            arr = np.load(path)
        elif path.endswith(".pkl"):
            with open(path, "rb") as f:
                arr = pickle.load(f)
        else:
            with open(path, "rb") as f:
                arr = pickle.load(f)
        arr = np.asarray(arr)
        return arr

    def generate_sequence_data(self):
        users = pd.read_pickle(self.config["data_path"])   # 원본 그대로

        cross = self._load_array(self.config["cross_emb_path"])  # (N, D)
        spe   = self._load_array(self.config["spe_emb_path"])    # (N, D)

        user_train = {}
        user_label = {}
        user_valid = {}

        for user in users:
            user_train[user] = users[user][-12:-2]
            user_label[user] = users[user][-2]
            user_valid[user] = users[user][-1]  # 마지막 아이템 예측

        return user_train, user_label, user_valid, cross, spe

    def get_train_valid_data(self):
        return self.user_train, self.user_label, self.user_valid, self.cross, self.spe

class BERTRecDataSet(Dataset):
    def __init__(self, user_train, max_len, num_user, item_min, num_item, mask_prob):
        self.user_train = user_train
        self.max_len = max_len
        self.num_user = num_user
        self.item_min = item_min
        self.num_item = num_item
        self.mask_prob = mask_prob
        self._all_items = set([i for i in range(self.item_min, self.num_item + 1)])
        self._all_items_len = len(self._all_items)

    def __len__(self):
        return self.num_user

    def __getitem__(self, user):
        user_seq = self.user_train[user]
        mask_len = self.max_len - len(user_seq)
        tokens = [0] * mask_len + user_seq
        return torch.LongTensor(tokens)

    def random_neg_sampling(self, rated_item : list, num_item_sample : int):
        nge_samples = random.sample(list(self._all_items - set(rated_item)), num_item_sample)
        return nge_samples
