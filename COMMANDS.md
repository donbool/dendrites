# DLL RNN Experiments - Command Reference Sheet

## Quick Start: Run All Experiments

### Option 1: Run Everything Automatically (Recommended)
```bash
# Run all 6 models with 15 epochs, then generate figures
./run_paper_experiments.sh
```

This will:
1. Clean old results
2. Train all 6 models (BP, DLL, Hierarchical, Temporal, Predictive, Bidirectional)
3. Generate all figures and tables
4. Save to `figures/` directory

---

## Option 2: Manual Step-by-Step

### Step 1: Activate Virtual Environment
```bash
source .venv/bin/activate
```

### Step 2: Clean Old Results (Important!)
```bash
rm -rf results/*.json results/*.pt figures/*.png
mkdir -p results figures
```

### Step 3: Train Individual Models

**Backprop Baseline:**
```bash
python3 experiments/run_exp.py \
    --run_bp \
    --epochs_bp 15 \
    --lr_bp 1e-2 \
    --batch_size 16 \
    --max_len 32
```

**Standard DLL:**
```bash
python3 experiments/run_exp.py \
    --run_dll \
    --epochs_dll 15 \
    --lr_dll 5e-4 \
    --batch_size 16 \
    --max_len 32
```

**Hierarchical DLL:**
```bash
python3 experiments/run_exp.py \
    --run_hierarchical \
    --epochs_hierarchical 15 \
    --lr_hierarchical 5e-4 \
    --batch_size 16 \
    --max_len 32
```

**Temporal DLL:**
```bash
python3 experiments/run_exp.py \
    --run_temporal \
    --epochs_temporal 15 \
    --lr_temporal 5e-4 \
    --batch_size 16 \
    --max_len 32
```

**Predictive Coding DLL:**
```bash
python3 experiments/run_exp.py \
    --run_predictive \
    --epochs_predictive 15 \
    --lr_predictive 5e-4 \
    --pred_weight 0.05 \
    --batch_size 16 \
    --max_len 32
```

**Bidirectional DLL:**
```bash
python3 experiments/run_exp.py \
    --run_bidirectional \
    --epochs_bidirectional 15 \
    --lr_bidirectional 5e-4 \
    --batch_size 16 \
    --max_len 32
```

### Step 4: Generate Figures
```bash
python3 experiments/visualize_results.py
```

---

## Option 3: Run Multiple Models at Once

```bash
source .venv/bin/activate

# Train multiple models in one command
python3 experiments/run_exp.py \
    --run_bp --run_dll --run_bidirectional \
    --epochs_bp 15 --epochs_dll 15 --epochs_bidirectional 15 \
    --batch_size 16 --max_len 32
```

---

## Quick Tests (Faster, for Debugging)

### Test Single Model (3 epochs, short sequences)
```bash
source .venv/bin/activate

python3 experiments/run_exp.py \
    --run_dll \
    --epochs_dll 3 \
    --batch_size 16 \
    --max_len 20
```

### Test Bidirectional vs Standard DLL
```bash
source .venv/bin/activate

python3 experiments/run_exp.py \
    --run_dll --run_bidirectional \
    --epochs_dll 5 --epochs_bidirectional 5 \
    --batch_size 16 --max_len 32
```

---

## Output Files

After running experiments, you'll have:

### Metrics (Raw Data)
```
results/
├── bp_YYYYMMDD_HHMMSS.json           # Backprop metrics
├── dll_YYYYMMDD_HHMMSS.json          # Standard DLL metrics
├── hierarchical_YYYYMMDD_HHMMSS.json # Hierarchical DLL metrics
├── temporal_YYYYMMDD_HHMMSS.json     # Temporal DLL metrics
├── predictive_YYYYMMDD_HHMMSS.json   # Predictive DLL metrics
├── bidirectional_YYYYMMDD_HHMMSS.json # Bidirectional DLL metrics
└── *.pt files (model weights)
```

### Figures (For Paper)
```
figures/
├── training_curves.png          # Training/validation curves (2x2 grid)
├── final_comparison.png         # Bar chart of final accuracy
├── dll_variants_only.png        # DLL variants comparison (2 plots)
├── convergence_speed.png        # How fast models converge
└── results_table.txt            # LaTeX + Markdown + CSV tables
```

---

## Hyperparameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs_MODEL` | 15 | Number of training epochs |
| `--lr_bp` | 1e-2 | Learning rate for backprop |
| `--lr_dll` | 5e-4 | Learning rate for DLL variants |
| `--batch_size` | 16 | Batch size (fixed for DLL) |
| `--max_len` | 32 | Maximum sequence length |
| `--pred_weight` | 0.05 | Prediction objective weight (Predictive DLL only) |
| `--temporal_decay` | 0.9 | Temporal error decay (Temporal DLL only) |

---

## Common Issues & Solutions

### Issue: "Batch size mismatch"
**Solution:** DLL requires fixed batch size. Use `--batch_size 16` (or 32) and ensure dataset uses `drop_last=True`.

### Issue: Visualization shows no data
**Solution:** Make sure you ran experiments first. Check `results/*.json` files exist.

### Issue: "ModuleNotFoundError: tabulate"
**Solution:** Install missing dependency:
```bash
source .venv/bin/activate
pip install tabulate
```

### Issue: Different sequence lengths in results
**Solution:** Clean old results and re-run with same `--max_len`:
```bash
rm -rf results/*.json
./run_paper_experiments.sh
```

---

## For Your Research Paper

### Run Final Experiments (Long, for Publication)
```bash
# Edit run_paper_experiments.sh to set EPOCHS=15 (or 20)
# Then:
./run_paper_experiments.sh
```

This takes ~30-60 minutes depending on hardware.

### Generate Only Figures (If Already Trained)
```bash
source .venv/bin/activate
python3 experiments/visualize_results.py
```

### Get Results Table
```bash
cat figures/results_table.txt
```

Copy the LaTeX table section directly into your paper's `.tex` file.

---

## Model Architecture Summary

| Model | Description | Bio-Plausible? | Key Feature |
|-------|-------------|----------------|-------------|
| **BP** | Standard backprop | ❌ | Optimal baseline |
| **DLL** | Dendritic local learning | ✅ | Local errors only |
| **Hierarchical** | Multi-timescale thetas | ✅ | 3 temporal zones |
| **Temporal** | Recurrent error paths | ✅ | Lateral dendrites |
| **Predictive** | Dual objectives (task + prediction) | ✅ | Auxiliary learning |
| **Bidirectional** | Forward + backward DLL cores | ✅ | Full context |

---

## Quick Reference: File Locations

```
dendrites/
├── models/
│   ├── bp_rnn.py                # Backprop baseline
│   ├── dll_rnn.py               # Standard DLL
│   ├── hierarchical_rnn.py      # Hierarchical DLL
│   ├── temporal_rnn.py          # Temporal DLL
│   ├── predictive_coding_rnn.py # Predictive DLL
│   └── bidirectional_dll_rnn.py # Bidirectional DLL
│
├── train/
│   ├── train_bp.py              # BP training loop
│   ├── train_dll.py             # DLL training loop
│   ├── train_hierarchical.py    # Hierarchical training
│   ├── train_temporal.py        # Temporal training
│   ├── train_predictive.py      # Predictive training
│   └── train_bidirectional.py   # Bidirectional training
│
├── experiments/
│   ├── run_exp.py               # Main experiment runner
│   ├── visualize_results.py     # Generate figures
│   └── run_full_experiments.py  # (Alternative runner)
│
├── run_paper_experiments.sh     # ⭐ ONE-CLICK FULL PIPELINE
│
├── results/                     # Metrics JSON files (auto-generated)
└── figures/                     # Plots and tables (auto-generated)
```

---

## Getting Help

- **Methods explanation**: See `METHODS_SECTION_GUIDE.md`
- **DLL theory**: See `TEMPORAL_DLL_EXPLANATION.md` and `PREDICTIVE_CODING_DLL_EXPLANATION.md`
- **Issues**: Check error messages, ensure virtual environment is activated

---

**TL;DR: Just run `./run_paper_experiments.sh` and come back in 30-60 minutes. All figures will be in `figures/`**
