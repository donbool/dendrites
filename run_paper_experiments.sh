#!/bin/bash
# Run all experiments with identical hyperparameters for fair comparison

# Clean old results
echo "Cleaning old results..."
rm -rf results/*.json
rm -rf results/*.pt
rm -rf figures/*.png
mkdir -p results
mkdir -p figures

# Experimental configuration (EDIT THESE)
EPOCHS=15
BATCH_SIZE=16
MAX_LEN=32
LR_DLL=5e-4
LR_BP=1e-2

echo "=========================================="
echo "Running experiments with:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Max sequence length: $MAX_LEN"
echo "  DLL learning rate: $LR_DLL"
echo "  BP learning rate: $LR_BP"
echo "=========================================="

# Activate virtual environment
source .venv/bin/activate

# Run each model sequentially with identical hyperparameters
echo ""
echo "========== 1/6: TRAINING BACKPROP BASELINE =========="
python3 experiments/run_exp.py \
    --run_bp \
    --epochs_bp $EPOCHS \
    --lr_bp $LR_BP \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN

echo ""
echo "========== 2/6: TRAINING STANDARD DLL =========="
python3 experiments/run_exp.py \
    --run_dll \
    --epochs_dll $EPOCHS \
    --lr_dll $LR_DLL \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN

echo ""
echo "========== 3/6: TRAINING HIERARCHICAL DLL =========="
python3 experiments/run_exp.py \
    --run_hierarchical \
    --epochs_hierarchical $EPOCHS \
    --lr_hierarchical $LR_DLL \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN

echo ""
echo "========== 4/6: TRAINING TEMPORAL DLL =========="
python3 experiments/run_exp.py \
    --run_temporal \
    --epochs_temporal $EPOCHS \
    --lr_temporal $LR_DLL \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN

echo ""
echo "========== 5/6: TRAINING PREDICTIVE CODING DLL =========="
python3 experiments/run_exp.py \
    --run_predictive \
    --epochs_predictive $EPOCHS \
    --lr_predictive $LR_DLL \
    --pred_weight 0.05 \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN

echo ""
echo "========== 6/6: TRAINING BIDIRECTIONAL DLL =========="
python3 experiments/run_exp.py \
    --run_bidirectional \
    --epochs_bidirectional $EPOCHS \
    --lr_bidirectional $LR_DLL \
    --batch_size $BATCH_SIZE \
    --max_len $MAX_LEN

echo ""
echo "=========================================="
echo "All training complete!"
echo "=========================================="

# Generate visualizations
echo ""
echo "Generating figures..."
python3 experiments/visualize_results.py

echo ""
echo "=========================================="
echo "✓ EXPERIMENTS COMPLETE"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - Metrics: ./results/"
echo "  - Figures: ./figures/"
echo ""
echo "Check figures/ directory for:"
echo "  - training_curves.png"
echo "  - final_comparison.png"
echo "  - dll_variants_only.png"
echo "  - convergence_speed.png"
echo "  - results_table.txt"
