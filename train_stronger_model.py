import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


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


def check_artifacts(paths):
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(missing))


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_epoch(model, data_tensor, optimizer, criterion, vocab_size, device, batch_size=64, train=True):
    model.train(train)
    order = torch.randperm(len(data_tensor)) if train else torch.arange(len(data_tensor))
    total_loss = 0.0
    steps = 0

    for i in range(0, len(data_tensor), batch_size):
        idx = order[i : i + batch_size]
        batch = data_tensor[idx].to(device)
        x = batch[:, :-1]
        y = batch[:, 1:]

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


def main():
    parser = argparse.ArgumentParser(description="Continue training baseline model to produce a stronger checkpoint.")
    parser.add_argument("--base-model", default="greedy_model.pth")
    parser.add_argument("--train-path", default="train.npy")
    parser.add_argument("--val-path", default="val.npy")
    parser.add_argument("--output-model", default="stronger_model.pth")
    parser.add_argument("--report", default="stronger_training_report.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    check_artifacts([args.base_model, args.train_path, args.val_path])
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_data = np.load(args.train_path)
    val_data = np.load(args.val_path)
    train_tensor = torch.tensor(train_data, dtype=torch.long)
    val_tensor = torch.tensor(val_data, dtype=torch.long)
    vocab_size = int(max(train_tensor.max().item(), val_tensor.max().item())) + 1

    model = DecoderOnlyTransformer(vocab_size).to(device)
    model.load_state_dict(torch.load(args.base_model, map_location=device))

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    history = []
    best_val_loss = float("inf")
    best_epoch = 0

    output_model_path = Path(args.output_model)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    initial_val_loss = run_epoch(
        model,
        val_tensor,
        optimizer,
        criterion,
        vocab_size,
        device,
        batch_size=args.batch_size,
        train=False,
    )
    initial_val_ppl = float(torch.exp(torch.tensor(initial_val_loss)).item())
    print(f"Initial Val Loss: {initial_val_loss:.4f} | PPL: {initial_val_ppl:.3f}")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_tensor,
            optimizer,
            criterion,
            vocab_size,
            device,
            batch_size=args.batch_size,
            train=True,
        )
        val_loss = run_epoch(
            model,
            val_tensor,
            optimizer,
            criterion,
            vocab_size,
            device,
            batch_size=args.batch_size,
            train=False,
        )
        train_ppl = float(torch.exp(torch.tensor(train_loss)).item())
        val_ppl = float(torch.exp(torch.tensor(val_loss)).item())

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), output_model_path)

        history.append(
            {
                "epoch": epoch,
                "train_loss": round(float(train_loss), 6),
                "train_ppl": round(float(train_ppl), 6),
                "val_loss": round(float(val_loss), 6),
                "val_ppl": round(float(val_ppl), 6),
                "is_best": bool(epoch == best_epoch),
            }
        )
        print(
            f"Epoch {epoch:02d} | Train Loss: {train_loss:.4f} | Train PPL: {train_ppl:.3f} | "
            f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.3f}"
        )

    payload = {
        "base_model": args.base_model,
        "output_model": str(output_model_path),
        "device": device,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "initial_val_loss": round(float(initial_val_loss), 6),
        "initial_val_ppl": round(float(initial_val_ppl), 6),
        "best_epoch": best_epoch,
        "best_val_loss": round(float(best_val_loss), 6),
        "best_val_ppl": round(float(torch.exp(torch.tensor(best_val_loss)).item()), 6),
        "history": history,
    }

    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved stronger checkpoint: {output_model_path}")
    print(f"Saved training report: {report_path}")


if __name__ == "__main__":
    main()
