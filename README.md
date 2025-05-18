# Arithmetic Transformer Project

A PyTorch-based implementation of a Transformer model for performing basic arithmetic operations (addition and subtraction) on character-level input sequences. This repository includes scripts for data generation, model training, evaluation, and analysis across multiple model versions.

---

## Repository Structure

```bash
.
├── create_datasets.py        # Generates and splits arithmetic problem datasets
├── data/                     # Generated CSV files:
│   ├── train.csv
│   ├── val.csv
│   ├── test.csv
│   └── generalization.csv
├── dataset.py                # ArithmeticDataset class & vocabulary creation
├── transformer.py            # Transformer model, masks, decoding utilities
├── train.py                  # Training script (includes dataset creation)
├── model_analysis.py         # Evaluation & comparison across model versions
├── requirements.txt          # Python dependencies
├── model/                    # Saved model checkpoints
└── results/                  # Generated prediction CSVs
```

---

## Requirements

* Python 3.8 or higher
* PyTorch 1.10+
* numpy
* tqdm

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you don’t have a `requirements.txt`, you can manually install:

```bash
pip install torch numpy tqdm
```

---

## Data Generation

All datasets are created via `create_datasets.py`.

```bash
python create_datasets.py
```

This script will write four CSV files under `data/`:

* `train.csv`        — Training split
* `val.csv`          — Validation split
* `test.csv`         — Test split
* `generalization.csv` — Out-of-distribution generalization split

---

## Training the Model

Run the training pipeline using `train.py`. By default, it:

1. Generates datasets (if not already present)
2. Builds DataLoaders with optimized batching
3. Instantiates a Transformer (`d_model=256, num_heads=8, num_layers=3, d_ff=1024`)
4. Trains for up to 50 epochs with early stopping (patience=3)
5. Saves the best checkpoint to `model/best_model.pt`

```bash
python train.py
```

To customize hyperparameters, open `train.py` and modify the `Transformer` arguments or training settings.

---

## Evaluating & Analyzing Models

Use `model_analysis.py` to:

* Load one or more saved model versions (`v3`, `v4`, `v5`)
* Evaluate on `test.csv` and `generalization.csv`
* Print metrics: Loss, Exact Match, Character Accuracy, Perplexity
* (Optional) Save per-example predictions to `results/`

```bash
python model_analysis.py --model all \
    [--test_dataset data/test.csv] \
    [--gen_dataset data/generalization.csv] \
    [--batch_size 128] \
    [--save_predictions] \
    [--results_dir results]
```

* `--model` : choose `v3`, `v4`, `v5`, or `all` (default: `all`)
* `--save_predictions` : writes `<version>_test_preds.csv` and `<version>_gen_preds.csv`

---

## Single-Example Decoding

Within `model_analysis.py`, the `decode_with_model()` helper uses greedy decoding to predict a single arithmetic problem:

```python
from transformer import Transformer
greedy_decode
def decode_single(problem: str, checkpoint: str):
    # load checkpoint, build vocab & model, then:
    result = greedy_decode(model, src_tensor, src_vocab, tgt_vocab)
    return result
```

Feel free to adapt this for an interactive REPL or web service.

---

## Creating `requirements.txt`

If missing, add:

```
torch
numpy
tqdm
```

---

## Directory Summary

* **data/**           Contains all CSV splits
* **model/**          Saved PyTorch checkpoint (`best_model.pt`)
* **results/**        Optional CSVs with predictions

---

Happy experimenting!

<!-- # Model Analysis Script

This script is designed to analyze and compare the performance of transformer models (V3, V4, and V5) trained for arithmetic tasks.

## Features

- Evaluate models on the test dataset
- Evaluate models on the generalization dataset (larger digit numbers)
- Test models on specific arithmetic problems
- Compare performance metrics across different models
- Identify the best-performing model based on exact match accuracy

## Requirements

- Python 3.6+
- PyTorch
- tqdm
- The `transformer_model.py` file must be in the same directory

## Usage

```bash
python model_analysis.py [--model MODEL] [--test_dataset TEST_DATASET] [--gen_dataset GEN_DATASET] [--batch_size BATCH_SIZE]
```

### Arguments

- `--model`: Model version to analyze. Options: 'v3', 'v4', 'v5', or 'all' (default: 'all')
- `--test_dataset`: Path to test dataset (default: 'data/test.csv')
- `--gen_dataset`: Path to generalization dataset (default: 'data/generalization.csv')
- `--batch_size`: Batch size for evaluation (default: 128)

### Example

To evaluate all models:
```bash
python model_analysis.py --model all
```

To evaluate only V3 model:
```bash
python model_analysis.py --model v3
```

## Model Versions

1. **V3 Model Parameters:**
   - d_model = 256
   - num_heads = 8
   - num_layers = 3
   - d_ff = 1024
   - dropout = 0.1
   - learning_rate = 0.00001

2. **V4 Model Parameters:**
   - d_model = 128
   - num_heads = 4
   - num_layers = 2
   - d_ff = 512
   - dropout = 0.1
   - learning_rate = 0.00001

3. **V5 Model Parameters:**
   - d_model = 256
   - num_heads = 8
   - num_layers = 4
   - d_ff = 2048
   - dropout = 0.1
   - learning_rate = 0.00001

## Output Metrics

The script will output the following metrics for each model:

- **Loss**: Cross-entropy loss value
- **Exact Match Accuracy**: The proportion of predictions that match the true answer exactly
- **Character Level Accuracy**: The proportion of correctly predicted characters
- **Perplexity**: Exponential of the average negative log-likelihood per token

## Comparison

The script will also compare the performance of the models and indicate which model performs best on the test and generalization datasets.  -->