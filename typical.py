import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class DecoderOnlyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=4, max_len=256):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def generate_square_subsequent_mask(self, size, device):
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask == 1, float("-inf"))

    def forward(self, x):
        _, t = x.size()
        mask = self.generate_square_subsequent_mask(t, x.device)
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer(x, mask=mask)
        return self.fc_out(x)


def load_vocab(c2i, i2c):
    with open(c2i, "rb") as f:
        char_to_idx = pickle.load(f)
    with open(i2c, "rb") as f:
        idx_to_char = pickle.load(f)
    return char_to_idx, idx_to_char


def typical_decode(model, start_text, char_to_idx, idx_to_char, device,
                   tau=0.9, max_length=500, max_context=256):

    if not (0 < tau <= 1):
        raise ValueError("tau must be in (0,1]")

    model.eval()
    generated = [char_to_idx[ch] for ch in start_text]
    input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)[0, -1]
            probs = torch.softmax(logits, dim=-1)

            log_probs = torch.log(probs + 1e-9)
            entropy = -(probs * log_probs).sum()

            # typicality = | -log(p) - entropy |
            typicality = torch.abs(-log_probs - entropy)

            sorted_typ, sorted_idx = torch.sort(typicality)
            sorted_probs = probs[sorted_idx]

            cumulative_probs = torch.cumsum(sorted_probs, dim=0)

            mask = cumulative_probs > tau
            mask[0] = False

            filtered_probs = sorted_probs.clone()
            filtered_probs[mask] = 0
            filtered_probs /= filtered_probs.sum()

            sampled_idx = torch.multinomial(filtered_probs, 1).item()
            next_token = sorted_idx[sampled_idx].item()

        generated.append(next_token)
        input_ids = torch.tensor([generated[-max_context:]], dtype=torch.long, device=device)

    return "".join(idx_to_char[i] for i in generated)



def compute_perplexity(model, data_tensor, vocab_size, device):
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    for seq in data_tensor:
        x = seq[:-1].unsqueeze(0).to(device)
        y = seq[1:].unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
        total_loss += loss.item()
    avg_loss = total_loss / len(data_tensor)
    return avg_loss, torch.exp(torch.tensor(avg_loss)).item()


def compute_ttr(text):
    return len(set(text)) / len(text) if text else 0.0


def shakespeare_line_score(text):
    lines = text.split("\n")
    valid = sum(1 for l in lines if 5 <= len(l.strip().split()) <= 12)
    return valid / max(len(lines), 1) * 100


def levenshtein_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def compute_cer(pred, target):
    return levenshtein_distance(pred, target) / max(len(target), 1)


def main():
    parser = argparse.ArgumentParser(description="Typical sampling evaluation")
    parser.add_argument("--model-path", default="greedy_model.pth")
    parser.add_argument("--val-path", default="val.npy")
    parser.add_argument("--char-to-idx", default="char_to_idx.pkl")
    parser.add_argument("--idx-to-char", default="idx_to_char.pkl")
    parser.add_argument("--tau-values", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--report", default="typical_report.json")
    parser.add_argument("--prompt", default="First Citizen: Before we proceed any further, hear me speak.")
    args = parser.parse_args()

    torch.manual_seed(42)
    np.random.seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    char_to_idx, idx_to_char = load_vocab(args.char_to_idx, args.idx_to_char)
    vocab_size = len(char_to_idx)

    model = DecoderOnlyTransformer(vocab_size).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    val_data = np.load(args.val_path)
    val_tensor = torch.tensor(val_data, dtype=torch.long)
    val_text = "".join(idx_to_char[int(i)] for seq in val_tensor for i in seq)

    val_loss, val_ppl = compute_perplexity(model, val_tensor, vocab_size, device)

    results = []

    for tau in args.tau_values:
        ttr, line, cer, samples = [], [], [], []

        for _ in range(args.runs):
            text = typical_decode(model, args.prompt, char_to_idx, idx_to_char, device, tau)
            samples.append(text)
            ttr.append(compute_ttr(text))
            line.append(shakespeare_line_score(text))
            cer.append(compute_cer(text[:len(val_text)], val_text[:len(text)]))

        results.append({
            "tau": tau,
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 4),
            "ttr_mean": round(float(np.mean(ttr)), 4),
            "line_score_mean": round(float(np.mean(line)), 4),
            "cer_mean": round(float(np.mean(cer)), 4),
            "sample_text": samples[0],
        })

    Path(args.report).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved report to {args.report}")


if __name__ == "__main__":
    main()