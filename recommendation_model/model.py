import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(ScaledDotProductAttention, self).__init__()
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, Q, K, V, mask):
        attn_score = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.hidden_units)
        attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_dist = self.dropout(F.softmax(attn_score, dim=-1))
        output = torch.matmul(attn_dist, V)
        return output, attn_dist

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_units = hidden_units
        self.W_Q = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_K = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_V = nn.Linear(hidden_units, hidden_units * num_heads, bias=False)
        self.W_O = nn.Linear(hidden_units * num_heads, hidden_units, bias=False)
        self.attention = ScaledDotProductAttention(hidden_units, dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6)

    def forward(self, enc, mask):
        residual = enc
        batch_size, seqlen = enc.size(0), enc.size(1)
        Q = self.W_Q(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)
        K = self.W_K(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)
        V = self.W_V(enc).view(batch_size, seqlen, self.num_heads, self.hidden_units)
        Q, K, V = Q.transpose(1, 2), K.transpose(1, 2), V.transpose(1, 2)
        output, attn_dist = self.attention(Q, K, V, mask)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seqlen, -1)
        output = self.layerNorm(self.dropout(self.W_O(output)) + residual)
        return output, attn_dist

class PositionwiseFeedForward(nn.Module):
    def __init__(self, hidden_units, dropout_rate):
        super(PositionwiseFeedForward, self).__init__()
        self.W_1 = nn.Linear(hidden_units, hidden_units)
        self.W_2 = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.layerNorm = nn.LayerNorm(hidden_units, 1e-6)

    def forward(self, x):
        residual = x
        output = self.W_2(F.relu(self.dropout(self.W_1(x))))
        output = self.layerNorm(self.dropout(output) + residual)
        return output

class BERT4RecBlock(nn.Module):
    def __init__(self, num_heads, hidden_units, dropout_rate):
        super(BERT4RecBlock, self).__init__()
        self.attention = MultiHeadAttention(num_heads, hidden_units, dropout_rate)
        self.pointwise_feedforward = PositionwiseFeedForward(hidden_units, dropout_rate)

    def forward(self, input_enc, mask):
        output_enc, attn_dist = self.attention(input_enc, mask)
        output_enc = self.pointwise_feedforward(output_enc)
        return output_enc, attn_dist

class BERT4Rec(nn.Module):
    def __init__(self, num_user, num_item, item_min, hidden_units, num_heads, num_layers, max_len, dropout_rate, device):
        super(BERT4Rec, self).__init__()
        self.num_user = num_user
        self.num_item = num_item
        self.item_min = item_min
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.device = device
        self.item_emb = nn.Embedding(num_item+2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.emb_layernorm = nn.LayerNorm(hidden_units, eps=1e-6)
        self.blocks = nn.ModuleList([BERT4RecBlock(num_heads, hidden_units, dropout_rate) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_units, 50)

    def forward(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.device))
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.device))
        seqs = self.emb_layernorm(self.dropout(seqs))
        mask_pad = torch.BoolTensor(log_seqs > 0).unsqueeze(1).repeat(1, log_seqs.shape[1], 1).unsqueeze(1).to(self.device)
        for block in self.blocks:
            seqs, attn_dist = block(seqs, mask_pad)
        out = self.out(seqs)
        return out

class CrossAttentionGating(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        self.hidden_units = hidden_units
        self.W1 = nn.Linear(hidden_units, hidden_units)
        self.W2 = nn.Linear(hidden_units, hidden_units)
        self.output_proj = nn.Linear(hidden_units, hidden_units)
        self.sigmoid = nn.Sigmoid()

    def forward(self, H, v1, v2):
        B, L, D = H.size()
        attn1 = torch.sum(H * v1.unsqueeze(1), dim=-1)  # (B, L)
        attn2 = torch.sum(H * v2.unsqueeze(1), dim=-1)  # (B, L)
        w1 = torch.softmax(attn1, dim=1).unsqueeze(-1)  # (B, L, 1)
        w2 = torch.softmax(attn2, dim=1).unsqueeze(-1)  # (B, L, 1)
        v1_exp = v1.unsqueeze(1).expand(B, L, D)
        v2_exp = v2.unsqueeze(1).expand(B, L, D)
        A1 = w1 * v1_exp
        A2 = w2 * v2_exp
        G = self.sigmoid(self.W1(A1) + self.W2(A2))     # (B, L, D)
        C = G * A1 + (1 - G) * A2
        C_pooled = torch.mean(C, dim=1)                 # (B, D)
        return C_pooled

class BERT4RecWithCrossAttention(nn.Module):
    def __init__(self, BERT4Rec,device):
        super(BERT4RecWithCrossAttention, self).__init__()
        self.bert4rec = BERT4Rec
        self.cross_attention_gating = CrossAttentionGating(50)
        self.item_embs = nn.Embedding(self.bert4rec.num_item+1,50)
        self.device = device

    def forward(self, log_seqs, vector1, vector2):
        bert4rec_output = self.bert4rec(log_seqs)
        combined_output = self.cross_attention_gating(bert4rec_output, vector1, vector2)
        num_items = self.bert4rec.num_item
        all_item_indices = torch.arange(0, num_items+1).to(self.device)
        item_embs = self.item_embs(all_item_indices).to(self.device)
        logits = torch.matmul(combined_output, item_embs.transpose(0, 1))
        return logits
