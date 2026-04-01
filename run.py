#!/usr/bin/env python3
"""
Interactive char-level LSTM chatbot runner.
Loads fine-tuned checkpoint and generates text with:
- sliding window context
- top-k sampling
- temperature control
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


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


@torch.no_grad()
def generate(
    model,
    prompt,
    char2idx,
    idx2char,
    max_new_chars=220,
    seq_len=80,
    temperature=0.4,
    top_k=8,
):
    model.eval()
    context = [char2idx.get(c, 0) for c in prompt]
    if not context:
        context = [0]

    out_chars = list(prompt)

    for _ in range(max_new_chars):
        x = torch.tensor([context[-seq_len:]], dtype=torch.long)
        logits, _ = model(x)
        next_logits = logits[0, -1] / max(temperature, 1e-6)

        k = min(top_k, next_logits.size(0)) if top_k > 0 else next_logits.size(0)
        vals, idxs = torch.topk(next_logits, k=k)
        probs = F.softmax(vals, dim=-1)
        next_local = torch.multinomial(probs, 1).item()
        next_id = idxs[next_local].item()

        context.append(next_id)
        ch = idx2char[next_id]
        out_chars.append(ch)

        if ch in "\n.!?" and len(out_chars) > len(prompt) + 40:
            break

    return "".join(out_chars)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default="checkpoints/fine_tuned_model.pt")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--max_new_chars", type=int, default=220)
    args = parser.parse_args()

    checkpoint = torch.load(args.ckpt, map_location="cpu")
    config = checkpoint["config"]
    char2idx = checkpoint["char2idx"]
    idx2char = {int(k): v for k, v in checkpoint["idx2char"].items()}

    model = CharLSTM(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config.get("dropout", 0.1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    seq_len = int(config.get("seq_len", 80))

    print("Loaded fine-tuned model. Type 'quit' to exit.")
    while True:
        user = input("\nYou: ").strip()
        if user.lower() in {"quit", "exit", "q"}:
            print("Bye.")
            break

        prompt = f"User: {user}\nAssistant: "
        response = generate(
            model,
            prompt,
            char2idx,
            idx2char,
            max_new_chars=args.max_new_chars,
            seq_len=seq_len,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        answer = response[len(prompt) :].strip()
        print(f"Assistant: {answer}")


if __name__ == "__main__":
    main()
