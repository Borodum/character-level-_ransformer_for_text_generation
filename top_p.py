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


def load_vocab(char_to_idx_path, idx_to_char_path):
    with open(char_to_idx_path, "rb") as f:
        char_to_idx = pickle.load(f)
    with open(idx_to_char_path, "rb") as f:
        idx_to_char = pickle.load(f)
    return char_to_idx, idx_to_char


def top_p_decode(
    model,
    start_text,
    char_to_idx,
    idx_to_char,
    device,
    p=0.9,
    max_length=500,
    max_context=256,
):
    if p <= 0.0 or p > 1.0:
        raise ValueError("p must be in (0, 1]")

    missing = [ch for ch in start_text if ch not in char_to_idx]
    if missing:
        raise ValueError(f"prompt contains characters outside vocabulary: {sorted(set(missing))}")

    model.eval()
    generated = [char_to_idx[ch] for ch in start_text]
    input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    for _ in range(max_length):
        with torch.no_grad():
            logits = model(input_ids)
            next_logits = logits[0, -1]
            probs = torch.softmax(next_logits, dim=-1)

            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = torch.searchsorted(cumulative_probs, torch.tensor(p, device=device)).item()
            cutoff = min(cutoff, sorted_probs.numel() - 1)

            nucleus_probs = sorted_probs[: cutoff + 1]
            nucleus_indices = sorted_indices[: cutoff + 1]
            nucleus_probs = nucleus_probs / nucleus_probs.sum()

            sampled_pos = torch.multinomial(nucleus_probs, num_samples=1).item()
            next_token = nucleus_indices[sampled_pos].item()

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
    ppl = torch.exp(torch.tensor(avg_loss)).item()
    return avg_loss, ppl


def compute_ttr(text):
    tokens = list(text)
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def shakespeare_line_score(text):
    lines = text.split("\n")
    valid_lines = 0
    for line in lines:
        words = line.strip().split()
        if 5 <= len(words) <= 12:
            valid_lines += 1
    return valid_lines / max(len(lines), 1) * 100


def levenshtein_distance(a, b):
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def compute_cer(pred_text, target_text):
    return levenshtein_distance(pred_text, target_text) / max(len(target_text), 1)


def check_artifacts(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def main():
    parser = argparse.ArgumentParser(description="Evaluate top-p sampling on char-level transformer.")
    parser.add_argument("--model-path", default="greedy_model.pth")
    parser.add_argument("--val-path", default="val.npy")
    parser.add_argument("--char-to-idx", default="char_to_idx.pkl")
    parser.add_argument("--idx-to-char", default="idx_to_char.pkl")
    parser.add_argument("--prompt", default="First Citizen: Before we proceed any further, hear me speak.")
    parser.add_argument("--max-length", type=int, default=500)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--p-values", type=float, nargs="+", default=[0.8, 0.9, 0.95])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", default="top_p_report.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    check_artifacts([args.model_path, args.val_path, args.char_to_idx, args.idx_to_char])

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
    print(f"Validation Loss: {val_loss:.4f} | Perplexity: {val_ppl:.3f}")

    results = []
    for p in args.p_values:
        ttr_scores = []
        line_scores = []
        cer_scores = []
        generated_samples = []

        for _ in range(args.runs):
            generated = top_p_decode(
                model,
                args.prompt,
                char_to_idx,
                idx_to_char,
                device,
                p=p,
                max_length=args.max_length,
            )
            generated_samples.append(generated)
            ttr_scores.append(compute_ttr(generated))
            line_scores.append(shakespeare_line_score(generated))
            cer_scores.append(compute_cer(generated[: len(val_text)], val_text[: len(generated)]))

        row = {
            "p": p,
            "val_loss": round(val_loss, 4),
            "val_ppl": round(val_ppl, 4),
            "ttr_mean": round(float(np.mean(ttr_scores)), 4),
            "line_score_mean": round(float(np.mean(line_scores)), 4),
            "cer_mean": round(float(np.mean(cer_scores)), 4),
            "sample_text": generated_samples[0],
        }
        results.append(row)

        print(f"\nTop-p={p:.2f}")
        print(f"TTR (mean): {row['ttr_mean']:.4f}")
        print(f"Shakespearean Line Structure Score (mean): {row['line_score_mean']:.2f}%")
        print(f"CER (mean): {row['cer_mean']:.4f}")
        print("Sample Generated Text:")
        print(generated_samples[0])

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved report to {args.report}")


if __name__ == "__main__":
    main()
