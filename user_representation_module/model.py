import torch
import torch.nn as nn
import torch.nn.functional as F

class BERT4Rec(nn.Module):
    def __init__(self, num_item, hidden_units, num_heads, num_layers, max_len, dropout_rate, device):
        super().__init__()
        self.item_emb = nn.Embedding(num_item + 2, hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_units)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_units, nhead=num_heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.layernorm = nn.LayerNorm(hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device

    def forward(self, seq):
        pos = torch.arange(seq.size(1), device=seq.device).unsqueeze(0)
        x = self.item_emb(seq) + self.pos_emb(pos)
        x = self.layernorm(self.dropout(x))
        x = self.encoder(x.transpose(0,1)).transpose(0,1)
        return x  # (B, L, D)

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x): return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output): return -grad_output

def grad_reverse(x): return GradReverse.apply(x)

class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x, grl=True):
        if grl: x = grad_reverse(x)
        return self.net(x)

def masked_loss(seq_output, target_seq, prediction_head):
    logits = prediction_head(seq_output)  # (B, L, V)
    logits = logits.view(-1, logits.size(-1))  # (B*L, V)
    target = target_seq.view(-1)  # (B*L)
    return F.cross_entropy(logits, target, ignore_index=0)

def domain_loss(preds, domain_labels):
    return F.cross_entropy(preds, domain_labels)

def alignment_loss(cross, common):
    return F.mse_loss(cross, common.detach())

def separation_loss(cross, specific, margin=1.0):
    dist = F.pairwise_distance(cross, specific)
    return F.relu(margin - dist).mean()
