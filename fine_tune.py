#!/usr/bin/env python3
"""
Fine-tune a base char-level LSTM checkpoint on dialogue text.
Python 3.8 compatible.
"""

import argparse
import json
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len=80, stride=1):
        encoded = [char2idx[c] for c in text if c in char2idx]
        self.inputs = []
        self.targets = []

        for i in range(0, len(encoded) - seq_len, stride):
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
    def __init__(self, vocab_size, embed_dim=64, hidden_size=128, num_layers=1, dropout=0.1):
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
        return self.fc(out), hidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_ckpt", default="checkpoints/base_model.pt")
    parser.add_argument("--dialogue_path", default="dialogue.txt")
    parser.add_argument("--out_dir", default="checkpoints")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seq_len", type=int, default=80)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    checkpoint = torch.load(args.base_ckpt, map_location="cpu")
    config = checkpoint["config"]
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]

    model = CharLSTM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.train()

    with open(args.dialogue_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    ds = CharDataset(text, char2idx=char2idx, seq_len=args.seq_len, stride=args.stride)
    if len(ds) == 0:
        raise ValueError("Fine-tune dataset has no usable chars from base vocabulary")

    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        for x, y in dl:
            logits, _ = model(x)
            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(1, len(dl))
        print(f"Fine-tune epoch {epoch}/{args.epochs} | loss={avg_loss:.4f}")

    out_ckpt = os.path.join(args.out_dir, "fine_tuned_model.pt")
    out_meta = os.path.join(args.out_dir, "fine_tuned_meta.json")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char2idx": char2idx,
            "idx2char": idx2char,
            "config": config,
        },
        out_ckpt,
    )

    with open(out_meta, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_checkpoint": args.base_ckpt,
                "fine_tuned_checkpoint": out_ckpt,
                "dialogue_chars": len(text),
                "fine_tune_samples": len(ds),
            },
            f,
            indent=2,
        )

    print(f"Saved fine-tuned checkpoint: {out_ckpt}")
    print(f"Saved metadata: {out_meta}")


if __name__ == "__main__":
    main()
