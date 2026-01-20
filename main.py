import random
import string
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# DATA GENERATION
# ============================================

class BijectionDataGenerator:
    def __init__(self, n):
        self.n = n
        self.set_a = list(range(n))
        self.set_b = list(range(n))
        self.b_to_letter = {i: string.ascii_uppercase[i] for i in range(min(n, 26))}
    
    def generate_bijection(self):
        shuffled_b = self.set_b.copy()
        random.shuffle(shuffled_b)
        return {a: b for a, b in zip(self.set_a, shuffled_b)}
    
    def display_b(self, b):
        return self.b_to_letter.get(b, str(b))
    
    def display_sequence(self, sequence):
        result = []
        for i, token in enumerate(sequence):
            if i % 2 == 0:
                result.append(str(token))
            else:
                result.append(self.display_b(token))
        return result
    
    def display_bijection(self, bijection):
        return {a: self.display_b(b) for a, b in bijection.items()}
    
    def display_posterior(self, posterior):
        return {self.display_b(i): f'{p:.2f}' for i, p in enumerate(posterior) if p > 0}
    
    def generate_training_example(self, bijection=None):
        if bijection is None:
            bijection = self.generate_bijection()
        
        reveal_order = self.set_a.copy()
        random.shuffle(reveal_order)
        
        sequence = []
        prediction_positions = []
        targets = []
        posteriors = []
        
        used_b = set()
        
        for a in reveal_order:
            b = bijection[a]
            
            remaining_b = set(self.set_b) - used_b
            posterior = [1.0 / len(remaining_b) if b_opt in remaining_b else 0.0 
                         for b_opt in self.set_b]
            
            sequence.append(a)
            prediction_positions.append(len(sequence))
            sequence.append(b)
            
            targets.append(b)
            posteriors.append(posterior)
            used_b.add(b)
        
        return {
            'sequence': sequence,
            'prediction_positions': prediction_positions,
            'targets': targets,
            'posteriors': posteriors,
            'bijection': bijection
        }


class BijectionDataset(Dataset):
    def __init__(self, n, num_bijections):
        self.generator = BijectionDataGenerator(n)
        self.n = n
        self.num_bijections = num_bijections
        self.vocab_size = 2 * n
        
    def __len__(self):
        return self.num_bijections * self.n
    
    def __getitem__(self, idx):
        example = self.generator.generate_training_example()
        step = idx % self.n
        
        full_seq = example['sequence']
        partial_seq = full_seq[:2*step + 1]
        target_b = example['targets'][step]
        posterior = example['posteriors'][step]
        
        tokens = []
        for i, val in enumerate(partial_seq):
            if i % 2 == 0:
                tokens.append(val)
            else:
                tokens.append(val + self.n)
        
        input_tokens = torch.tensor(tokens, dtype=torch.long)
        target = torch.tensor(target_b + self.n, dtype=torch.long)
        posterior_tensor = torch.tensor(posterior, dtype=torch.float)
        
        return {
            'input_tokens': input_tokens,
            'target': target,
            'posterior': posterior_tensor,
            'step': step
        }


def collate_fn(batch):
    max_len = max(item['input_tokens'].size(0) for item in batch)
    
    padded_inputs = []
    targets = []
    posteriors = []
    lengths = []
    steps = []
    
    for item in batch:
        seq = item['input_tokens']
        seq_len = seq.size(0)
        lengths.append(seq_len)
        
        padded = torch.zeros(max_len, dtype=torch.long)
        padded[:seq_len] = seq
        padded_inputs.append(padded)
        
        targets.append(item['target'])
        posteriors.append(item['posterior'])
        steps.append(item['step'])
    
    return {
        'input_tokens': torch.stack(padded_inputs),
        'targets': torch.stack(targets),
        'posteriors': torch.stack(posteriors),
        'lengths': torch.tensor(lengths),
        'steps': torch.tensor(steps)
    }


# ============================================
# TRANSFORMER MODEL
# ============================================

class BijectionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(d_model, vocab_size)
    
    def forward(self, x, lengths=None):
        seq_len = x.size(1)
        
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        padding_mask = None
        if lengths is not None:
            padding_mask = torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        tok_emb = self.embedding(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        h = tok_emb + pos_emb
        
        h = self.transformer(h, mask=causal_mask, src_key_padding_mask=padding_mask)
        
        logits = self.output_head(h)
        
        return logits


# ============================================
# MLP MODEL
# ============================================

class BijectionMLP(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model=64, num_layers=4, hidden_dim=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(512, d_model)
        
        self.input_proj = nn.Linear(max_seq_len * d_model, hidden_dim)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            ))
        
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.output_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, lengths=None):
        batch_size, seq_len = x.shape
        
        tok_emb = self.embedding(x)
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos)
        h = tok_emb + pos_emb
        
        if seq_len < self.max_seq_len:
            padding = torch.zeros(batch_size, self.max_seq_len - seq_len, self.d_model, device=x.device)
            h = torch.cat([h, padding], dim=1)
        
        h = h.view(batch_size, -1)
        h = self.input_proj(h)
        
        for layer in self.layers:
            h = h + layer(h)
        
        h = self.final_norm(h)
        logits = self.output_head(h)
        
        return logits


# ============================================
# RNN (LSTM) MODEL
# ============================================

class BijectionRNN(nn.Module):
    def __init__(self, vocab_size, d_model=64, hidden_dim=128, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.output_head = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, lengths=None):
        batch_size, seq_len = x.shape
        
        # Embed tokens
        h = self.embedding(x)  # (batch, seq_len, d_model)
        
        # Pack padded sequence for efficient LSTM processing
        if lengths is not None:
            h = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # LSTM forward
        lstm_out, (hidden, cell) = self.lstm(h)
        
        # Unpack if packed
        if lengths is not None:
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)
        
        # Output logits for all positions
        logits = self.output_head(lstm_out)  # (batch, seq_len, vocab_size)
        
        return logits


# ============================================
# TRAINING FUNCTIONS
# ============================================

def train_epoch_transformer(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        input_tokens = batch['input_tokens'].to(device)
        targets = batch['targets'].to(device)
        lengths = batch['lengths'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_tokens, lengths)
        
        batch_size = input_tokens.size(0)
        last_pos_logits = logits[torch.arange(batch_size), lengths - 1]
        
        loss = F.cross_entropy(last_pos_logits, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def train_epoch_mlp(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        input_tokens = batch['input_tokens'].to(device)
        targets = batch['targets'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_tokens)
        
        loss = F.cross_entropy(logits, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def train_epoch_rnn(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in dataloader:
        input_tokens = batch['input_tokens'].to(device)
        targets = batch['targets'].to(device)
        lengths = batch['lengths'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(input_tokens, lengths)
        
        # Get prediction at last position for each sequence
        batch_size = input_tokens.size(0)
        last_pos_logits = logits[torch.arange(batch_size), lengths - 1]
        
        loss = F.cross_entropy(last_pos_logits, targets)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


# ============================================
# EVALUATION
# ============================================

def evaluate_entropy_by_step(model, n, device, num_samples=500, model_type='transformer'):
    """Compute average model entropy at each step and compare to Bayesian entropy."""
    model.eval()
    gen = BijectionDataGenerator(n)
    
    model_entropies = {k: [] for k in range(n)}
    bayesian_entropies = {k: np.log2(n - k) for k in range(n)}
    
    with torch.no_grad():
        for _ in range(num_samples):
            example = gen.generate_training_example()
            
            for step in range(n):
                partial_seq = example['sequence'][:2*step + 1]
                
                tokens = []
                for i, val in enumerate(partial_seq):
                    if i % 2 == 0:
                        tokens.append(val)
                    else:
                        tokens.append(val + n)
                
                input_tensor = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)
                length_tensor = torch.tensor([len(tokens)]).to(device)
                
                if model_type == 'transformer':
                    logits = model(input_tensor, length_tensor)
                    last_logits = logits[0, -1, n:]
                elif model_type == 'mlp':
                    logits = model(input_tensor)
                    last_logits = logits[0, n:]
                elif model_type == 'rnn':
                    logits = model(input_tensor, length_tensor)
                    last_logits = logits[0, -1, n:]
                
                probs = F.softmax(last_logits, dim=-1).cpu().numpy()
                
                probs = probs + 1e-10
                entropy = -np.sum(probs * np.log2(probs))
                model_entropies[step].append(entropy)
    
    avg_model_entropies = [np.mean(model_entropies[k]) for k in range(n)]
    bayesian = [bayesian_entropies[k] for k in range(n)]
    
    return avg_model_entropies, bayesian


def compute_mae(model_entropies, bayesian_entropies):
    return np.mean(np.abs(np.array(model_entropies) - np.array(bayesian_entropies)))


# ============================================
# MAIN
# ============================================

def main():
    # Hyperparameters
    N = 20  # Larger bijection size
    VOCAB_SIZE = 2 * N
    MAX_SEQ_LEN = 2 * N
    
    # Model hyperparameters
    D_MODEL = 64
    NHEAD = 4
    NUM_LAYERS_TRANSFORMER = 4
    DIM_FF = 256
    
    NUM_LAYERS_MLP = 6
    HIDDEN_DIM_MLP = 256
    
    NUM_LAYERS_RNN = 3
    HIDDEN_DIM_RNN = 128
    
    BATCH_SIZE = 64
    NUM_TRAIN = 5000
    EPOCHS = 50
    LR = 1e-3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Bijection size N = {N}")
    print(f"Training for {EPOCHS} epochs")
    
    # Create models
    transformer = BijectionTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS_TRANSFORMER,
        dim_feedforward=DIM_FF
    ).to(device)
    
    mlp = BijectionMLP(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS_MLP,
        hidden_dim=HIDDEN_DIM_MLP
    ).to(device)
    
    rnn = BijectionRNN(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        hidden_dim=HIDDEN_DIM_RNN,
        num_layers=NUM_LAYERS_RNN
    ).to(device)
    
    transformer_params = sum(p.numel() for p in transformer.parameters())
    mlp_params = sum(p.numel() for p in mlp.parameters())
    rnn_params = sum(p.numel() for p in rnn.parameters())
    
    print(f"\nModel parameters:")
    print(f"  Transformer: {transformer_params:,}")
    print(f"  MLP:         {mlp_params:,}")
    print(f"  RNN (LSTM):  {rnn_params:,}")
    
    # Optimizers
    optimizer_transformer = torch.optim.AdamW(transformer.parameters(), lr=LR, weight_decay=0.01)
    optimizer_mlp = torch.optim.AdamW(mlp.parameters(), lr=LR, weight_decay=0.01)
    optimizer_rnn = torch.optim.AdamW(rnn.parameters(), lr=LR, weight_decay=0.01)
    
    # Training history
    transformer_losses = []
    mlp_losses = []
    rnn_losses = []
    
    print("\nTraining:")
    print("-" * 70)
    print(f"{'Epoch':>6} | {'Transformer':>14} | {'MLP':>14} | {'RNN':>14}")
    print("-" * 70)
    
    for epoch in range(EPOCHS):
        # Create fresh dataloaders each epoch
        train_dataset = BijectionDataset(N, NUM_TRAIN)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        
        t_loss = train_epoch_transformer(transformer, train_loader, optimizer_transformer, device)
        
        train_dataset_mlp = BijectionDataset(N, NUM_TRAIN)
        train_loader_mlp = DataLoader(train_dataset_mlp, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        m_loss = train_epoch_mlp(mlp, train_loader_mlp, optimizer_mlp, device)
        
        train_dataset_rnn = BijectionDataset(N, NUM_TRAIN)
        train_loader_rnn = DataLoader(train_dataset_rnn, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        r_loss = train_epoch_rnn(rnn, train_loader_rnn, optimizer_rnn, device)
        
        transformer_losses.append(t_loss)
        mlp_losses.append(m_loss)
        rnn_losses.append(r_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"{epoch+1:>6} | {t_loss:>14.4f} | {m_loss:>14.4f} | {r_loss:>14.4f}")
    
    # Evaluate entropy curves
    print("\nEvaluating entropy curves (this may take a minute)...")
    transformer_entropies, bayesian = evaluate_entropy_by_step(transformer, N, device, num_samples=300, model_type='transformer')
    mlp_entropies, _ = evaluate_entropy_by_step(mlp, N, device, num_samples=300, model_type='mlp')
    rnn_entropies, _ = evaluate_entropy_by_step(rnn, N, device, num_samples=300, model_type='rnn')
    
    # Compute MAE
    transformer_mae = compute_mae(transformer_entropies, bayesian)
    mlp_mae = compute_mae(mlp_entropies, bayesian)
    rnn_mae = compute_mae(rnn_entropies, bayesian)
    
    print(f"\nResults (Mean Absolute Error from Bayesian):")
    print(f"  Transformer: {transformer_mae:.4f} bits")
    print(f"  MLP:         {mlp_mae:.4f} bits")
    print(f"  RNN (LSTM):  {rnn_mae:.4f} bits")
    
    # ============================================
    # PLOTTING
    # ============================================
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Training loss curves
    ax1 = axes[0]
    ax1.plot(transformer_losses, label='Transformer', color='blue')
    ax1.plot(mlp_losses, label='MLP', color='red')
    ax1.plot(rnn_losses, label='RNN', color='green')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Cross-Entropy Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Entropy curves
    ax2 = axes[1]
    steps = list(range(1, N + 1))
    ax2.plot(steps, bayesian, 'k--', label='Bayesian (true)', linewidth=2)
    ax2.plot(steps, transformer_entropies, 'b-o', label=f'Transformer (MAE={transformer_mae:.3f})', markersize=3)
    ax2.plot(steps, mlp_entropies, 'r-s', label=f'MLP (MAE={mlp_mae:.3f})', markersize=3)
    ax2.plot(steps, rnn_entropies, 'g-^', label=f'RNN (MAE={rnn_mae:.3f})', markersize=3)
    ax2.set_xlabel('Step k')
    ax2.set_ylabel('Entropy (bits)')
    ax2.set_title('Predictive Entropy vs Bayesian Entropy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Error bars
    ax3 = axes[2]
    transformer_error = np.array(transformer_entropies) - np.array(bayesian)
    mlp_error = np.array(mlp_entropies) - np.array(bayesian)
    rnn_error = np.array(rnn_entropies) - np.array(bayesian)
    
    width = 0.25
    x = np.array(steps)
    ax3.bar(x - width, transformer_error, width=width, label='Transformer', color='blue', alpha=0.7)
    ax3.bar(x, mlp_error, width=width, label='MLP', color='red', alpha=0.7)
    ax3.bar(x + width, rnn_error, width=width, label='RNN', color='green', alpha=0.7)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Step k')
    ax3.set_ylabel('Entropy Error (bits)')
    ax3.set_title('Deviation from Bayesian Entropy')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformer_vs_mlp_vs_rnn_N20.png', dpi=150)
    print("\nSaved plot to 'transformer_vs_mlp_vs_rnn_N20.png'")
    plt.show()


if __name__ == "__main__":
    main()