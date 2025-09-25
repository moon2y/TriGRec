import numpy as np
import torch
from torch.utils.data import Dataset

class BERTRecDataSet(Dataset):
    def __init__(self, user_train, max_len, num_user, num_item, mask_prob):
        self.user_train = user_train
        self.max_len = max_len
        self.num_user = num_user
        self.num_item = num_item
        self.mask_prob = mask_prob

    def __len__(self):
        return self.num_user

    def __getitem__(self, user):
        user_seq = self.user_train[user]
        tokens, labels = [], []
        masked_at_least_one = False

        for s in user_seq[-self.max_len:]:
            prob = np.random.random()
            if prob < self.mask_prob:
                tokens.append(self.num_item + 1)
                labels.append(s)
                masked_at_least_one = True
            else:
                tokens.append(s)
                labels.append(0)

        if not masked_at_least_one:
            random_index = np.random.randint(len(user_seq[-self.max_len:]))
            tokens[random_index] = self.num_item + 1
            labels[random_index] = user_seq[-self.max_len:][random_index]

        pad_len = self.max_len - len(tokens)
        tokens = [0]*pad_len + tokens
        labels = [0]*pad_len + labels

        return torch.LongTensor(tokens), torch.LongTensor(labels)

class InferenceDataset(Dataset):
    def __init__(self, user_dict, max_len, num_item):
        self.users = list(user_dict.keys())
        self.user_dict = user_dict
        self.max_len = max_len
        self.num_item = num_item

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        uid = self.users[idx]
        seq = self.user_dict[uid]
        seq = seq[-self.max_len:]
        pad_len = self.max_len - len(seq)
        tokens = [0]*pad_len + seq
        return uid, torch.LongTensor(tokens)
