# dendritic localized learning vs backpropagation on POS tagging

can biologically plausible dendritic localized learning (DLL) match standard backpropagation on sequence tagging tasks?

## proj overview

this project implements and compares two learning algorithms:
- **BP baseline**: standard RNN with backpropagation + adam optimizer
- **DLL**: biologically plausible three-factor learning rule from Lv et al. (2025)

both models train on Penn Treebank POS tagging (sequence labeling) with identical architecture and hyperparameters. ONLY the learning rule differs.

## architecture

- **embedding**: token → embed_dim (150)
- **RNN**: tanh-based recurrent layer (hidden_size=128)
- **tagger**: per-token classifier at each timestep → POS tags (~45 classes)

## how to run

```bash
# install dependencies
pip install -r requirements.txt

# train BP baseline
python experiments/run_exp.py --run_bp --epochs_bp 10

# train DLL model
python experiments/run_exp.py --run_dll --epochs_dll 10

# train both
python experiments/run_exp.py --run_bp --run_dll --epochs_bp 10 --epochs_dll 10
```

## hyperparameters

- `batch_size`: 32
- `max_len`: 50 (sequence length)
- `embed_dim`: 150
- `hidden_size`: 128
- `lr_bp`: 1e-2 (Adam learning rate)
- `lr_dll`: 5e-4 (DLL weight learning rate)

## core files

- [models/bp_rnn.py](models/bp_rnn.py): Standard PyTorch RNN for sequence tagging
- [models/dll_rnn.py](models/dll_rnn.py): DLL core engine + wrapper for sequence tagging
- [train/train_bp.py](train/train_bp.py): BP training loop
- [train/train_dll.py](train/train_dll.py): DLL training loop (no backprop, local updates only)
- [data/dataset_pos_tagging.py](data/dataset_pos_tagging.py): Penn Treebank POS dataset loader

## main ref

- Lv et al. (2025): *Dendritic Localized Learning* ([arxiv](https://arxiv.org/html/2501.09976v1))
- OG repo: https://github.com/Lvchangze/Dendritic-Localized-Learning

---
COMS6998: Computation & the Brain
