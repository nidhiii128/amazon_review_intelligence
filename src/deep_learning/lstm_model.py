"""
Shared pieces of the LSTM sentiment classifier: tokenizer, vocabulary,
the BiLSTM-with-attention network, and a small predict() helper. Kept in
its own module so the training script and the dashboard's live classifier
use exactly the same preprocessing and architecture.

Architecture:
  Embedding (trained from scratch) -> 2-layer bidirectional LSTM ->
  attention pooling over time steps -> dropout -> Linear -> 3 classes
Attention pooling lets the model weight the words that matter most
(e.g. "terrible", "love") instead of relying only on the final hidden
state, which matters for long reviews.
"""
import re
from collections import Counter

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

LABELS = ["negative", "neutral", "positive"]
PAD, UNK = "<pad>", "<unk>"
TOKEN_RE = re.compile(r"[a-z0-9']+|[!?]")

DEFAULT_CONFIG = {
    "max_vocab": 30000, "min_freq": 3, "max_len": 200,
    "embed_dim": 128, "hidden_dim": 128, "num_layers": 2, "dropout": 0.4,
}


def tokenize(text: str):
    return TOKEN_RE.findall(str(text).lower())


def build_vocab(texts, max_vocab: int, min_freq: int) -> dict:
    counts = Counter(tok for t in texts for tok in tokenize(t))
    words = [w for w, c in counts.most_common(max_vocab - 2) if c >= min_freq]
    vocab = {PAD: 0, UNK: 1}
    for w in words:
        vocab[w] = len(vocab)
    return vocab


def encode(text: str, vocab: dict, max_len: int):
    ids = [vocab.get(tok, 1) for tok in tokenize(text)][:max_len]
    length = max(len(ids), 1)
    ids = ids or [1]
    return ids + [0] * (max_len - len(ids)), length


class BiLSTMAttention(nn.Module):
    def __init__(self, vocab_size: int, embed_dim=128, hidden_dim=128,
                 num_layers=2, dropout=0.4, num_classes=3, pooling="attention"):
        super().__init__()
        self.pooling = pooling  # "attention" (default) or "mean" (ablation baseline)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, lengths, return_attention=False):
        emb = self.dropout(self.embedding(x))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        if self.pooling == "mean":
            mask = (x != 0).unsqueeze(-1).float()
            context = (out * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            logits = self.fc(self.dropout(context))
            return (logits, mask.squeeze(-1) / mask.sum(1).clamp(min=1.0)) if return_attention else logits
        scores = self.attn(out).squeeze(-1)
        scores = scores.masked_fill(x == 0, float("-inf"))
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)
        logits = self.fc(self.dropout(context))
        return (logits, weights) if return_attention else logits


def load_checkpoint(path, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = BiLSTMAttention(
        len(ckpt["vocab"]), cfg["embed_dim"], cfg["hidden_dim"], cfg["num_layers"], cfg["dropout"],
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["vocab"], cfg


@torch.no_grad()
def predict(model, vocab, cfg, texts, device="cpu", with_attention=False):
    """Returns (probabilities [n,3], optional per-token attention list)."""
    ids, lens = zip(*(encode(t, vocab, cfg["max_len"]) for t in texts))
    x = torch.tensor(ids, device=device)
    lengths = torch.tensor(lens)
    logits, weights = model(x, lengths, return_attention=True)
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    if not with_attention:
        return probs, None
    attn = []
    for text, w, n in zip(texts, weights.cpu().numpy(), lens):
        toks = tokenize(text)[: cfg["max_len"]] or [UNK]
        attn.append(list(zip(toks, w[:n].tolist())))
    return probs, attn
