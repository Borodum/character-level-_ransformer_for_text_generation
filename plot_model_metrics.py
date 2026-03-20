import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


METRICS = [
    ("ttr", "Type-Token Ratio (TTR)"),
    ("line_score", "Shakespearean Line Structure Score"),
    ("cer", "Character Error Rate (CER)"),
]

METHOD_ORDER = ["Greedy", "Beam Search", "Temperature", "Top-k", "Top-p"]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pick_best(rows, cer_key):
    return min(rows, key=lambda row: row[cer_key])


def method_summary(model_reports_dir):
    model_reports_dir = Path(model_reports_dir)
    beam_rows = load_json(model_reports_dir / "beam_search_report.json")
    temperature_rows = load_json(model_reports_dir / "temperature_scaling_report.json")
    top_k_rows = load_json(model_reports_dir / "top_k_report.json")
    top_p_rows = load_json(model_reports_dir / "top_p_report.json")

    greedy_row = None
    beam_candidates = []
    for row in beam_rows:
        if int(row["beam_width"]) == 1:
            greedy_row = row
        else:
            beam_candidates.append(row)

    if greedy_row is None:
        raise ValueError("beam_search_report.json must include beam_width=1 to represent greedy baseline.")
    if not beam_candidates:
        raise ValueError("beam_search_report.json must include at least one beam_width>1 for beam search.")

    beam_row = pick_best(beam_candidates, "cer")
    temp_row = pick_best(temperature_rows, "cer_mean")
    top_k_row = pick_best(top_k_rows, "cer_mean")
    top_p_row = pick_best(top_p_rows, "cer_mean")

    summary = {
        "Greedy": {
            "ttr": float(greedy_row["ttr"]),
            "line_score": float(greedy_row["line_score"]),
            "cer": float(greedy_row["cer"]),
            "source": {"beam_width": int(greedy_row["beam_width"])},
        },
        "Beam Search": {
            "ttr": float(beam_row["ttr"]),
            "line_score": float(beam_row["line_score"]),
            "cer": float(beam_row["cer"]),
            "source": {"beam_width": int(beam_row["beam_width"])},
        },
        "Temperature": {
            "ttr": float(temp_row["ttr_mean"]),
            "line_score": float(temp_row["line_score_mean"]),
            "cer": float(temp_row["cer_mean"]),
            "source": {"temperature": float(temp_row["temperature"])},
        },
        "Top-k": {
            "ttr": float(top_k_row["ttr_mean"]),
            "line_score": float(top_k_row["line_score_mean"]),
            "cer": float(top_k_row["cer_mean"]),
            "source": {"k": int(top_k_row["k"])},
        },
        "Top-p": {
            "ttr": float(top_p_row["ttr_mean"]),
            "line_score": float(top_p_row["line_score_mean"]),
            "cer": float(top_p_row["cer_mean"]),
            "source": {"p": float(top_p_row["p"])},
        },
    }
    return summary


def plot_metric(summary, metric_key, metric_label, model_name, output_path):
    methods = [m for m in METHOD_ORDER if m in summary]
    values = [summary[m][metric_key] for m in methods]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(methods, values, color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"])
    ax.set_title(f"{model_name}: {metric_label}")
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Method")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def generate_for_model(model_name, reports_dir, out_dir):
    summary = method_summary(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, metric_label in METRICS:
        plot_metric(
            summary=summary,
            metric_key=metric_key,
            metric_label=metric_label,
            model_name=model_name,
            output_path=out_dir / f"{metric_key}.png",
        )

    (out_dir / "selected_method_points.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Build metric charts for baseline and stronger models.")
    parser.add_argument("--baseline-reports", default="reports/baseline")
    parser.add_argument("--stronger-reports", default="reports/stronger")
    parser.add_argument("--out-dir", default="plots")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    generate_for_model("Baseline Model", Path(args.baseline_reports), out_dir / "baseline")
    generate_for_model("Stronger Model", Path(args.stronger_reports), out_dir / "stronger")
    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
