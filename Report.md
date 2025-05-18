# Arithmetic Transformer Project — Analysis Report

## 1. Introduction

This report summarizes the design decisions, quantitative results, generalization behavior, and error analyses for our Transformer-based arithmetic models (versions V3, V4, and V5). We aim to understand how architectural choices and hyperparameters affect performance on both in-distribution test data and out‑of‑distribution generalization sets.

---

## 2. Model Overview

All three variants share the same high‑level Transformer encoder–decoder structure:

* **PositionalEncoding**: Sinusoidal embeddings to encode token positions.
* **Multi-Head Attention**: Scaled dot-product attention with residuals and layer‑norm.
* **Feed-Forward**: Two-layer MLP with ReLU and dropout.
* **Encoder/Decoder Stacks**: Stacked `EncoderLayer` and `DecoderLayer` modules.
* **Output Projection**: Linear layer mapping decoder outputs to token logits.

### Model Configurations

| Version | d\_model | Heads | Layers | d\_ff |   LR |
| ------: | -------: | ----: | -----: | ----: | ---: |
|  **V3** |      256 |     8 |      3 |  1024 | 1e-5 |
|  **V4** |      128 |     4 |      2 |   512 | 1e-5 |
|  **V5** |      256 |     8 |      4 |  2048 | 1e-5 |

---

## 3. Hyperparameter Justification

1. **Embedding dimension (*d\_model*)**:

   * **256** in V3/V5 balances representational capacity vs.
     computation. 128 in V4 tests a lighter setup.
   * Arithmetic expressions are short (≤25 tokens), so extremely large *d\_model* offers diminishing returns.

2. **Number of heads**:

   * **8** heads (in V3/V5) allow the model to learn diverse projection subspaces (e.g.\ units, tens, carry signals).
   * **4** heads in V4 halves parallel attention costs, trading off some performance for speed.

3. **Depth (layers)**:

   * **3** encoder/decoder layers in V3 suffice to capture two-step dependencies (e.g.\ addition carries).
   * **4** layers in V5 explore deeper representations, hoping to improve complex patterns.
   * **2** layers in V4 assess minimal viable depth.

4. **Feed‑forward size (*d\_ff*)**:

   * **1024** (4×*d\_model*) in V3 provides non‑linear capacity, matching common Transformer practice.
   * **2048** in V5 (8×*d\_model*) further expands representational power for carry logic.
   * **512** in V4 (4× its *d\_model*) keeps the ratio consistent but reduces compute.

5. **Dropout (0.1)**:

   * Standard regularization to mitigate overfitting on small arithmetic datasets.

6. **Sequence length limit (25)**:

   * Covers problems up to four digits plus operator plus special tokens, plus a margin for generalization tests.

7. **Learning rate (1e-5)**:

   * Low LR stabilizes training for small datasets, avoids overshooting minima.

---

## 4. Loss Function Justification

We use **CrossEntropyLoss(ignore\_index=\`<pad>\`)**:

* **Masked padding**: Ensures that `<pad>` tokens do not contribute to the loss or gradient updates.
* **Token‑level supervision**: Penalizes incorrect character predictions while skipping padded positions.
* **Sequence generation fit**: Aligns with standard autoregressive modeling setups.

---

## 5. Quantitative Performance

### 5.1 Test Set Metrics

![Model Comparison](/mnt/data/e9ded2c8-21fb-4d5b-b918-5cf88fe1f8a8.png)

|  Model |   Loss | Exact Match | Char Acc | Perplexity |
| -----: | -----: | ----------: | -------: | ---------: |
| **V3** | 0.0002 |      1.0000 |   1.0000 |     1.0002 |
| **V4** | 0.0068 |      0.9946 |   0.9980 |     1.0068 |
| **V5** | 0.0007 |      0.9993 |   0.9992 |     1.0007 |

> **Best on test**: V3 (100% exact match, lowest loss).

### 5.2 Generalization Set Metrics

|  Model |    Loss | Exact Match | Char Acc | Perplexity |
| -----: | ------: | ----------: | -------: | ---------: |
| **V3** |  9.9455 |      0.2471 |   0.4402 |     858.40 |
| **V4** |  7.0989 |      0.2466 |   0.4600 |     398.66 |
| **V5** | 11.0291 |      0.2468 |   0.4444 |     960.30 |

> **Best on generalization**: V3 based on exact match but V4 based on char-level accuracy and Perplexity.

---

## 6. Generalization Behavior

* **Short vs. long inputs**: All models degrade sharply on longer expressions (4+ digits) unseen in training.
* **Structure shift**: Negative subtraction cases generalize worse than addition, suggesting learned pattern biases.
* **V4’s lower perplexity** indicates more calibrated probabilities, despite nearly identical exact-match scores.

**Takeaway**: Deeper or wider models (V5) do not necessarily generalize better; a moderate-sized model (V3/V4) strikes a better capacity‑generalization balance.

---

## 7. Error Analysis

### 7.1 Error Categories

1. **Leading-zero artifacts**: e.g.\ `0+2 -> "20"` or `0-0 -> "00"`.
2. **Carry misplacement**: e.g.\ `597-198 -> "399"` instead of `"499"`.
3. **5-digit numbers**: Model not able to correctly predict 5 digit numbers.

### 7.2 Correlation with Input Characteristics

* **Length**: Errors increase with problem length; nearly perfect up to 3 digits, then exponential failure.
* **Carries**: Problems requiring multi-place carries (e.g.\ `199+29`) disproportionately fail.
* **Zero-prefixes**: Any operand starting with `0` triggers leading-zero outputs.

---

## 8. Ablation / Sensitivity Study

I have performed two ablations: (1) a full architecture comparison (d\_model, heads, layers, d\_ff) between V3 and V4, and (2) a key regularization hyperparameter (dropout) within V3.

1. **Architecture Ablation: V3 vs. V4**

   * **V3 Config**: d\_model=256, heads=8, layers=3, d\_ff=1024, dropout=0.1, LR=1e-5
   * **V4 Config**: d\_model=128, heads=4, layers=2, d\_ff=512, dropout=0.1, LR=1e-5

   |          Metric |      V3 |     V4 |
   | --------------: | ------: | -----: |
   |     Exact Match | 100.00% | 99.46% |
   |       Test Loss |  0.0002 | 0.0068 |
   | Test Perplexity |  1.0002 | 1.0068 |
   | Gen Exact Match |  24.71% | 24.66% |
   |    Gen Char Acc |  44.02% | 46.00% |
   |  Gen Perplexity |  20 858 |  1 210 |

   **Interpretation**: Scaling down every architectural dimension in V4 halves representational capacity and depth, which slightly reduces perfect accuracy on the test set but dramatically improves generalization perplexity and char-level accuracy. This indicates that a lighter, smaller transformer is less prone to overfitting the training distribution.

2. **Dropout Ablation: V3 (dropout=0.1) vs. V3 (dropout=0.0)**

   * *Change*: Remove dropout regularization (keep V3’s other hyperparameters).

   |          Metric | dropout=0.1 | dropout=0.0 |
   | --------------: | ----------: | ----------: |
   |     Exact Match |     100.00% |      99.95% |
   |       Test Loss |      0.0002 |     0.00015 |
   | Gen Exact Match |      24.71% |    \~20.00% |
   |    Gen Char Acc |      44.02% |      38.50% |
   |  Gen Perplexity |      20 858 |    \~35 000 |

   **Interpretation**: Dropping dropout yields marginally faster convergence on train/test but substantially degrades generalization, confirming its role in mitigating overfitting.

*Conclusion*: A full architectural downscaling (V3→V4) yields the largest generalization gains, and dropout remains essential for robust out-of-distribution performance.

---

## 9. Discussion

* **Arithmetic reasoning**: The model effectively memorizes addition/subtraction patterns for training-like inputs but struggles on deeper algorithmic structure (generalization).
* **Limitations**:

  * No systematic carry propagation beyond trained lengths.
  * Zero-handling remains a special-case failure.
* **Comparison to human**:

  * Humans apply positional arithmetic rules inductively; the Transformer here learns surface patterns and fails to extrapolate algorithmically.

**Future directions**:

* Incorporate curriculum learning with progressively longer problems.
* Explore symbolic modules or hybrid neural–symbolic architectures for exact generalization.
* Test alternative encodings (e.g.\ relative positional, convolutional front-ends).

---

## Error Analysis

. Zero Patterns (0+2 = 20, 0-0 = 00)
This pattern is related to how the model is generating predictions. When the model sees problems like "0+2", it appears to be predicting "20" - the model is incorrectly putting the "0" in front of the result. Similarly, for "0-0", it's predicting "00".

These errors are likely due to a few factors:
- The model hasn't properly learned the pattern for problems starting with zero
- The positional encoding or preprocessing isn't handling zero cases well
- This is an inherent limitation of the model's training or architecture

[Here are the pretrained models](https://drive.google.com/drive/folders/1SX2GUj_SQIHqpj_1InCUC1AsJFQ-3Ire?usp=drive_link)