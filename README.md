# character-level-_ransformer_for_text_generation
## Comparing 8 sampling methods to get clear, reusable insights into when and how to use each method.
####	Problem: Autoregressive language models rely on decoding strategies to generate text. While sampling methods and hyperparameters (temperature, top-k, top-p) critically control the diversity-coherence trade-off, they lack systematic, controlled comparison—especially on character-level models. Furthermore, advanced methods like Mirostat and Typical Sampling remain underexplored in small-scale, controlled testbeds.
####	Objective: To implement a character-level autoregressive Transformer and conduct a controlled comparative study of eight distinct sampling strategies:
-	Greedy Search (Deterministic baseline)
-	Beam Search (Width = 3, 5)
-	Pure Random Sampling (Ancestral)	
-	Temperature Scaling
-	Top-k Sampling
-	Top-p (Nucleus) Sampling
-	Typical Sampling
-	Mirostat (Adaptive dynamic sampling)
####	Research Question: How do these eight sampling strategies compare in managing the trade-off between output diversity and textual coherence in character-level text generation? Specifically, which methods produce the most "Shakespearean-like" text while avoiding the pitfalls of repetition (greedy methods) or incoherence (high-temperature methods)?
####	Sampling methods strongly affect generative models but lack systematic study. Character-level models offer a simple testbed. This research will give clear, reusable insights into when and how to use each method.

## Temperature Scaling Experiment

This branch adds a dedicated temperature-scaling evaluation that mirrors the greedy baseline metrics:
- Validation Loss
- Perplexity (PPL)
- Type-Token Ratio (TTR)
- Shakespearean Line Structure Score
- Character Error Rate (CER)

### Files
- `temperature_scaling.ipynb`: notebook workflow for interactive runs
- `temperature_scaling.py`: CLI script for repeatable runs

### Run (CLI)
Required artifacts (from preprocessing + baseline training):
- `val.npy`
- `char_to_idx.pkl`
- `idx_to_char.pkl`
- `greedy_model.pth`

```bash
python temperature_scaling.py \
  --model-path greedy_model.pth \
  --val-path val.npy \
  --char-to-idx char_to_idx.pkl \
  --idx-to-char idx_to_char.pkl \
  --temperatures 0.7 1.0 1.3 \
  --runs 5 \
  --max-length 500
```

The script saves aggregated results to `temperature_scaling_report.json`.

## Beam Search Experiment

This branch also includes beam-search decoding evaluation with the same metrics:
- Validation Loss
- Perplexity (PPL)
- Type-Token Ratio (TTR)
- Shakespearean Line Structure Score
- Character Error Rate (CER)

### Files
- `beam_search.ipynb`: notebook workflow
- `beam_search.py`: CLI script

### Run (PowerShell)
Required artifacts:
- `val.npy`
- `char_to_idx.pkl`
- `idx_to_char.pkl`
- `greedy_model.pth`

```powershell
python beam_search.py `
  --model-path greedy_model.pth `
  --val-path val.npy `
  --char-to-idx char_to_idx.pkl `
  --idx-to-char idx_to_char.pkl `
  --beam-widths 3 5 `
  --max-length 500
```

The script saves results to `beam_search_report.json`.
<<<<<<< HEAD
<<<<<<< Updated upstream
=======
=======
>>>>>>> cda39567cff2c44cdbad03c17a9496a7bbbe51ef

## Top-k Experiment

This branch also includes top-k sampling evaluation with the same metrics:

### Files
- `top_k.ipynb`: notebook workflow
- `top_k.py`: CLI script

### Run (PowerShell)
Required artifacts:
- `val.npy`
- `char_to_idx.pkl`
- `idx_to_char.pkl`
- `greedy_model.pth`

```powershell
python top_k.py `
  --model-path greedy_model.pth `
  --val-path val.npy `
  --char-to-idx char_to_idx.pkl `
  --idx-to-char idx_to_char.pkl `
  --k-values 10 40 100 `
  --runs 5 `
  --max-length 500
```

The script saves results to `top_k_report.json`.

<<<<<<< HEAD
## Top-p Experiment

This repository now also includes top-p (nucleus) sampling evaluation with the same metrics:

### Files
- `top_p.py`: CLI script

### Run (PowerShell)
=======
## Top-p (Nucleus) Sampling Experiment

This experiment evaluates top-p (nucleus) sampling using the same evaluation pipeline as the baseline.

### Files
- `top_p.ipynb`: notebook workflow
- `top_p.py`: CLI script

### Run (PowerShell)

>>>>>>> cda39567cff2c44cdbad03c17a9496a7bbbe51ef
```powershell
python top_p.py `
  --model-path greedy_model.pth `
  --val-path val.npy `
  --char-to-idx char_to_idx.pkl `
  --idx-to-char idx_to_char.pkl `
  --p-values 0.8 0.9 0.95 `
  --runs 5 `
  --max-length 500
<<<<<<< HEAD
```

The script saves results to `top_p_report.json`.

## Stronger Model + Comparison Plots

To build a stronger checkpoint (same architecture, continued training), re-run all methods for both models, and generate metric plots:

### Files
- `train_stronger_model.py`: continues training from `greedy_model.pth` and saves `stronger_model.pth`
- `plot_model_metrics.py`: builds 3 metric charts per model (TTR, Line Score, CER)

### Example Workflow (PowerShell)
```powershell
# 1) Train stronger checkpoint
python train_stronger_model.py `
  --base-model greedy_model.pth `
  --train-path train.npy `
  --val-path val.npy `
  --output-model stronger_model.pth `
  --report reports/stronger/training_report.json `
  --epochs 25 `
  --batch-size 64 `
  --lr 1e-4

# 2) Re-run methods for baseline model
python beam_search.py --model-path greedy_model.pth --beam-widths 1 3 5 --report reports/baseline/beam_search_report.json
python temperature_scaling.py --model-path greedy_model.pth --report reports/baseline/temperature_scaling_report.json
python top_k.py --model-path greedy_model.pth --report reports/baseline/top_k_report.json
python top_p.py --model-path greedy_model.pth --report reports/baseline/top_p_report.json

# 3) Re-run methods for stronger model
python beam_search.py --model-path stronger_model.pth --beam-widths 1 3 5 --report reports/stronger/beam_search_report.json
python temperature_scaling.py --model-path stronger_model.pth --report reports/stronger/temperature_scaling_report.json
python top_k.py --model-path stronger_model.pth --report reports/stronger/top_k_report.json
python top_p.py --model-path stronger_model.pth --report reports/stronger/top_p_report.json

# 4) Build plots
python plot_model_metrics.py `
  --baseline-reports reports/baseline `
  --stronger-reports reports/stronger `
  --out-dir plots
```

Generated plots:
- `plots/baseline/ttr.png`
- `plots/baseline/line_score.png`
- `plots/baseline/cer.png`
- `plots/stronger/ttr.png`
- `plots/stronger/line_score.png`
- `plots/stronger/cer.png`
>>>>>>> Stashed changes
=======
  ```

The script saves results to `top_p_report.json`.

## Pure Random (Ancestral) Sampling Experiment

This experiment evaluates pure random (ancestral) sampling using the same evaluation pipeline as previous methods.

### Files
- `rand_samp.ipynb`: notebook workflow
- `rand_samp.py`: CLI script

### Run (PowerShell)

```powershell
python rand_samp.py `
  --model-path greedy_model.pth `
  --val-path val.npy `
  --char-to-idx char_to_idx.pkl `
  --idx-to-char idx_to_char.pkl `
  --runs 5 `
  --max-length 500
```

The script saves results to `rand_samp_report.json`.

## Typical Sampling Experiment

This experiment evaluates typical sampling, which selects tokens based on their deviation from the expected information content (entropy).

### Files
- `typical.ipynb`: notebook workflow
- `typical.py`: CLI script

### Run (PowerShell)

```powershell
python typical.py `
  --model-path greedy_model.pth `
  --val-path val.npy `
  --char-to-idx char_to_idx.pkl `
  --idx-to-char idx_to_char.pkl `
  --tau-values 0.8 0.9 0.95 `
  --runs 5 `
  --max-length 500
```

The script saves results to `typical_report.json`.

## Mirostat Sampling Experiment

This experiment evaluates Mirostat, an adaptive sampling method that dynamically controls generation entropy (surprise) during decoding.

### Files
- `mirostat.ipynb`: notebook workflow  
- `mirostat.py`: CLI script  

### Run (PowerShell)

```powershell
python mirostat.py `
  --model-path greedy_model.pth `
  --val-path val.npy `
  --char-to-idx char_to_idx.pkl `
  --idx-to-char idx_to_char.pkl `
  --tau-values 3.0 5.0 7.0 `
  --eta 0.5 `
  --runs 5 `
  --max-length 500
```
The script saves results to `mirostat_report.json`.
>>>>>>> cda39567cff2c44cdbad03c17a9496a7bbbe51ef
