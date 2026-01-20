# Bijection Learning Experiment

> [!NOTE]
> Code was mostly vibe-coded with claude

This experiment is a replication of the **bijection learning task** from:

> **"The Bayesian Geometry of Transformer Attention"**  
> Naman Aggarwal, Siddhartha R. Dalal, Vishal Misra  
> arXiv:2512.22471 (December 2025)  
> https://arxiv.org/abs/2512.22471

The original paper introduces "Bayesian wind tunnels" — controlled environments where the true Bayesian posterior is known analytically and memorization is provably impossible. They show that small transformers reproduce Bayesian posteriors with 10⁻³–10⁻⁴ bit accuracy, while capacity-matched MLPs fail by orders of magnitude.

I replicate the bijection elimination task and add an **RNN (LSTM) baseline** to the comparison.

## The Task

Given a bijection (one-to-one mapping) between two sets A and B of size N:

- Elements are revealed one pair at a time in random order
- At each step, the model must predict the next B element given an A element
- The key insight: as more pairs are revealed, fewer options remain

**Example with N=4:**

```
A = {0, 1, 2, 3}
B = {A, B, C, D}

Step 1: See "2" → Predict from {A,B,C,D} → 4 options (entropy = 2 bits)
Step 2: See "2→C, 0" → Predict from {A,B,D} → 3 options (entropy = 1.58 bits)
Step 3: See "2→C, 0→A, 3" → Predict from {B,D} → 2 options (entropy = 1 bit)
Step 4: See "2→C, 0→A, 3→D, 1" → Predict from {B} → 1 option (entropy = 0 bits)
```

## What We're Measuring

A **Bayesian optimal** predictor assigns uniform probability over remaining valid options. Its entropy at step k is:

```
H(k) = log₂(N - k)
```

I measure how closely each model's predictive entropy matches this Bayesian ideal using Mean Absolute Error (MAE).

## Models Compared

| Model           | Description                                  |
| --------------- | -------------------------------------------- |
| **Transformer** | Decoder-only with causal masking (GPT-style) |
| **MLP**         | Feedforward network on flattened sequence    |
| **RNN**         | LSTM processing sequence autoregressively    |

## Results

![Results](transformer_vs_mlp_vs_rnn_N20.png)

| Model           | MAE from Bayesian (bits) |
| --------------- | ------------------------ |
| **RNN (LSTM)**  | 0.003                    |
| **Transformer** | 0.096                    |
| **MLP**         | 0.462                    |

**Surprising finding:** The RNN outperformed the Transformer on this task!

- **RNN** nearly perfectly matches the Bayesian optimal curve (MAE = 0.003)
- **Transformer** does well but has slight deviations (MAE = 0.096)
- **MLP** fails to track the decreasing entropy, especially at later steps where it becomes overconfident (negative error = entropy too low)

This differs from the original paper's results where Transformers achieved ~0.003 bit accuracy. Possible explanations:

1. My Transformer may need more training or hyperparameter tuning
2. The LSTM's recurrent state is well-suited for this elimination-tracking task
3. The task size (N=20) may favor RNN's sequential processing

## Code Structure

```
bijection_experiment.py
├── BijectionDataGenerator    # Generates random bijections and training sequences
├── BijectionDataset          # PyTorch Dataset with variable-length sequences
├── BijectionTransformer      # Decoder-only transformer (encoder + causal mask)
├── BijectionMLP              # Baseline MLP
├── BijectionRNN              # Baseline LSTM
├── Training functions        # Per-architecture training loops
├── Evaluation                # Entropy computation and comparison to Bayesian
└── main()                    # Runs experiment and generates plots
```

## Usage

```bash
python main.py
```

### Hyperparameters

| Parameter              | Default | Description                        |
| ---------------------- | ------- | ---------------------------------- |
| N                      | 20      | Bijection size (\|A\| = \|B\| = N) |
| EPOCHS                 | 50      | Training epochs                    |
| BATCH_SIZE             | 64      | Batch size                         |
| D_MODEL                | 64      | Embedding dimension                |
| NUM_LAYERS_TRANSFORMER | 4       | Transformer depth                  |
| NUM_LAYERS_RNN         | 3       | LSTM depth                         |

## Technical Notes

### Vocabulary encoding

Tokens from set A use indices `0` to `N-1`, tokens from set B use indices `N` to `2N-1`. This separation helps the model distinguish between "query" and "answer" tokens.

### Sequence format

```
[a₁, b₁, a₂, b₂, ..., aₖ]
         ↑
    predict bₖ here
```

The model predicts at odd positions (after seeing an A element).

## Requirements

- PyTorch
- NumPy
- Matplotlib

## References

- Aggarwal, N., Dalal, S. R., & Misra, V. (2025). _The Bayesian Geometry of Transformer Attention_. arXiv:2512.22471. https://arxiv.org/abs/2512.22471
