#!/usr/bin/env python3
"""
Minimal char-level LSTM base trainer (CPU-friendly).
Python 3.8 compatible.
"""

import argparse
import json
import os
import random
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class CharDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 80, stride: int = 1):
        self.seq_len = seq_len
        self.stride = stride

        chars = sorted(list(set(text)))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}

        encoded = [self.char2idx[c] for c in text]
        self.inputs = []
        self.targets = []

        last = len(encoded) - seq_len
        for i in range(0, last, stride):
            x = encoded[i : i + seq_len]
            y = encoded[i + 1 : i + seq_len + 1]
            if len(y) == seq_len:
                self.inputs.append(x)
                self.targets.append(y)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.inputs[idx], dtype=torch.long),
            torch.tensor(self.targets[idx], dtype=torch.long),
        )


class CharLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden


@torch.no_grad()
def sample_text(model, seed, char2idx, idx2char, length=200, temperature=0.4, top_k=8):
    model.eval()
    device = next(model.parameters()).device
    context = [char2idx.get(c, 0) for c in seed]
    if not context:
        context = [0]

    generated = list(seed)
    window = min(100, max(50, len(context)))

    for _ in range(length):
        x = torch.tensor([context[-window:]], dtype=torch.long, device=device)
        logits, _ = model(x)
        next_logits = logits[0, -1, :] / max(temperature, 1e-6)

        if top_k > 0:
            vals, idxs = torch.topk(next_logits, k=min(top_k, next_logits.size(-1)))
            probs = F.softmax(vals, dim=-1)
            pick = torch.multinomial(probs, 1).item()
            next_id = idxs[pick].item()
        else:
            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1).item()

        context.append(next_id)
        generated.append(idx2char[next_id])

    return "".join(generated)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def save_meta(path: str, meta: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="tiny_textbooks.txt")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--seq_len", type=int, default=80)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_chars", type=int, default=0, help="Optional cap for faster runs")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    text = load_text(args.data_path)
    if args.max_chars and args.max_chars > 0:
        text = text[: args.max_chars]

    if len(text) < args.seq_len + 1:
        raise ValueError("Dataset too small for chosen seq_len")

    ds = CharDataset(text, seq_len=args.seq_len, stride=args.stride)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = CharLSTM(
        vocab_size=len(ds.char2idx),
        embed_dim=args.embed_dim,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    print(f"Vocab size: {len(ds.char2idx)}")
    print(f"Train samples: {len(ds)}")
    print(f"Trainable params: {count_parameters(model):,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dl))
        print(f"Epoch {epoch}/{args.epochs} | loss={avg_loss:.4f}")

    elapsed = time.time() - t0
    print(f"Training finished in {elapsed/60:.2f} min")

    ckpt_path = os.path.join(args.out_dir, "base_model.pt")
    meta_path = os.path.join(args.out_dir, "base_meta.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char2idx": ds.char2idx,
            "idx2char": ds.idx2char,
            "config": {
                "vocab_size": len(ds.char2idx),
                "embed_dim": args.embed_dim,
                "hidden_size": args.hidden_size,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
                "seq_len": args.seq_len,
            },
        },
        ckpt_path,
    )
    save_meta(
        meta_path,
        {
            "checkpoint": ckpt_path,
            "elapsed_minutes": round(elapsed / 60, 3),
            "dataset_chars": len(text),
            "train_samples": len(ds),
        },
    )

    print(f"Saved base checkpoint: {ckpt_path}")
    print(f"Saved metadata: {meta_path}")

    seed = "The "
    sample = sample_text(model, seed, ds.char2idx, ds.idx2char, length=200)
    print("\nSample generation:\n")
    print(sample)


if __name__ == "__main__":
    main()
