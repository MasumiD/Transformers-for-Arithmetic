import random
import os
import csv
from typing import Tuple, List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm
import math
import torch.nn.utils.rnn as rnn_utils
from create_datasets import create_datasets, save_dataset

datasets = create_datasets()

# Save each split
save_dataset(datasets["train"], "train.csv")
save_dataset(datasets["val"], "val.csv")
save_dataset(datasets["test"], "test.csv")
save_dataset(datasets["generalization"], "generalization.csv")

print("Datasets written to data/train.csv, val.csv, test.csv and generalization.csv")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure positional encoding is on the same device as input
        return x + self.pe[:, :x.size(1)].to(x.device)

class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self attention with residual connection and layer norm
        # Convert mask from shape [batch_size, 1, 1, seq_len] to [batch_size, seq_len]
        # for key_padding_mask in MultiheadAttention
        if mask is not None:
            key_padding_mask = ~mask.squeeze(1).squeeze(1)
        else:
            key_padding_mask = None
        
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, enc_output: torch.Tensor, 
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        # Convert masks to the format expected by MultiheadAttention
        if src_mask is not None:
            src_key_padding_mask = ~src_mask.squeeze(1).squeeze(1)
        else:
            src_key_padding_mask = None
        
        # For self-attention in decoder, we need to handle the causal masking
        if tgt_mask is not None:
            # Create a causal attention mask (seq_len x seq_len)
            seq_len = x.size(1)
            # PyTorch expects attn_mask with False for positions to attend to
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            
            # Extract padding information from tgt_mask
            # For key_padding_mask, True indicates positions to ignore
            if tgt_mask.dim() == 3:  # [batch_size, tgt_len, tgt_len] (batch x tgt_len x tgt_len)
                # Extract the padding information from the mask
                tgt_key_padding_mask = ~tgt_mask[:, 0, :].bool()  # [batch_size, tgt_len]
            else:
                tgt_key_padding_mask = None
        else:
            causal_mask = None
            tgt_key_padding_mask = None
        
        # Self attention with residual connection and layer norm
        attn_output, _ = self.self_attn(
            x, x, x,
            attn_mask=causal_mask,
            key_padding_mask=tgt_key_padding_mask
        )
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Cross attention with residual connection and layer norm
        attn_output, _ = self.cross_attn(
            x, enc_output, enc_output,
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm2(x + self.dropout2(attn_output))
        
        # Feed forward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))
        
        return x

class Transformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 512,
                 num_heads: int = 8, num_layers: int = 6, d_ff: int = 2048,
                 dropout: float = 0.1, max_len: int = 25):
        super().__init__()
        
        # Save model dimensions
        self.d_model = d_model
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Encoder and Decoder stacks
        self.encoder = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.decoder = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer
        self.fc = nn.Linear(d_model, tgt_vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize parameters
        self._init_parameters()
        
    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor = None, tgt_mask: torch.Tensor = None) -> torch.Tensor:
        # Source embeddings
        src = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        
        # Target embeddings
        tgt = self.dropout(self.pos_encoding(self.tgt_embedding(tgt)))
        
        # Encoder
        enc_output = src
        for enc_layer in self.encoder:
            enc_output = enc_layer(enc_output, src_mask)
            
        # Decoder
        dec_output = tgt
        for dec_layer in self.decoder:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
            
        # Output
        output = self.fc(dec_output)
        
        return output 
    
class ArithmeticDataset(Dataset):
    def __init__(self, data_path: str, src_vocab: Dict[str, int], tgt_vocab: Dict[str, int]):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'
        
        # Read data from CSV
        with open(data_path, 'r') as f:
            lines = f.readlines()[1:]  # Skip header
            self.data = [line.strip().split(',') for line in lines]
            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        problem, solution = self.data[idx]
        
        # Convert to token indices
        src_tokens = [self.src_vocab[self.sos_token]] + \
                    [self.src_vocab.get(c, self.src_vocab[self.pad_token]) for c in problem] + \
                    [self.src_vocab[self.eos_token]]
        tgt_tokens = [self.tgt_vocab[self.sos_token]] + \
                    [self.tgt_vocab.get(c, self.tgt_vocab[self.pad_token]) for c in solution] + \
                    [self.tgt_vocab[self.eos_token]]
        
        return torch.tensor(src_tokens), torch.tensor(tgt_tokens)

def create_masks(src: torch.Tensor, tgt: torch.Tensor, pad_token: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create masks for transformer attention.
    
    Returns:
        src_mask: Padding mask for source sequence. Shape [batch_size, 1, 1, src_len]
        tgt_mask: Combined causal and padding mask for target sequence. Shape [batch_size, tgt_len, tgt_len]
    """
    device = src.device
    
    # Source mask (padding) - [batch_size, 1, 1, src_len]
    # 1 indicates positions to attend to, 0 indicates positions to ignore
    src_mask = (src != pad_token).unsqueeze(1).unsqueeze(2)
    
    # Target mask (combination of padding and causal masking)
    tgt_len = tgt.size(1)
    batch_size = tgt.size(0)
    
    # Create causal mask (lower triangular) - [tgt_len, tgt_len]
    # 1 indicates positions to attend to, 0 indicates positions to ignore
    causal_mask = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1) == 0
    
    # Expand causal mask to batch dimension - [batch_size, tgt_len, tgt_len]
    causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)
    
    # Create padding mask - [batch_size, tgt_len]
    # 1 indicates valid token positions, 0 indicates padding
    padding_mask = (tgt != pad_token)
    
    # Convert padding mask to attention mask shape - [batch_size, tgt_len, tgt_len]
    # Each position in target can attend to all non-padding positions
    padding_attn_mask = padding_mask.unsqueeze(1).expand(-1, tgt_len, -1)
    
    # Combine masks: can only attend if both masks allow it
    # 1 indicates positions to attend to, 0 indicates positions to ignore
    tgt_mask = causal_mask & padding_attn_mask
    
    return src_mask, tgt_mask

class PadCollate:
    def __init__(self, src_pad_idx: int, tgt_pad_idx: int):
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
    
    def __call__(self, batch):
        src_batch, tgt_batch = zip(*batch)
        # lengths
        src_lens = [t.size(0) for t in src_batch]
        tgt_lens = [t.size(0) for t in tgt_batch]
        max_src = max(src_lens)
        max_tgt = max(tgt_lens)

        # allocate
        padded_src = torch.full((len(batch), max_src), self.src_pad_idx, dtype=torch.long)
        padded_tgt = torch.full((len(batch), max_tgt), self.tgt_pad_idx, dtype=torch.long)

        # copy
        for i, (src, tgt) in enumerate(batch):
            padded_src[i, : src.size(0)] = src
            padded_tgt[i, : tgt.size(0)] = tgt

        return padded_src, padded_tgt

class Trainer:
    def __init__(self, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
                 learning_rate: float = 0.0001, device: str = 'cuda'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Loss function (ignoring padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=train_loader.dataset.tgt_vocab['<pad>'])
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        
        # Training state
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0
        
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0
        total_tokens = 0
        
        for src, tgt in tqdm(self.train_loader, desc='Training'):
            # Move data to device
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt, self.train_loader.dataset.src_vocab['<pad>'])
            
            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1])
            
            # Compute loss
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Calculate metrics
            non_pad_mask = (tgt[:, 1:] != self.train_loader.dataset.tgt_vocab['<pad>'])
            total_loss += loss.item() * non_pad_mask.sum().item()
            total_tokens += non_pad_mask.sum().item()
        
        return total_loss / total_tokens
    
    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float, float, float]:
        self.model.eval()
        total_loss = 0
        total_tokens = 0
        exact_matches = 0
        char_accuracy = 0
        total_samples = 0
        
        for src, tgt in tqdm(self.val_loader, desc='Evaluating'):
            # Move data to device
            src, tgt = src.to(self.device), tgt.to(self.device)
            
            # Create masks
            src_mask, tgt_mask = create_masks(src, tgt, self.val_loader.dataset.src_vocab['<pad>'])
            
            # Forward pass
            output = self.model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1])
            
            # Compute loss
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt[:, 1:].reshape(-1))
            
            # Calculate metrics
            non_pad_mask = (tgt[:, 1:] != self.val_loader.dataset.tgt_vocab['<pad>'])
            total_loss += loss.item() * non_pad_mask.sum().item()
            total_tokens += non_pad_mask.sum().item()
            
            # Calculate accuracy metrics
            predictions = output.argmax(dim=-1)
            for i in range(len(predictions)):
                pred_seq = predictions[i]
                true_seq = tgt[i, 1:]
                
                # Exact match
                if torch.equal(pred_seq, true_seq):
                    exact_matches += 1
                
                # Character accuracy
                mask = true_seq != self.val_loader.dataset.tgt_vocab['<pad>']
                if mask.any():
                    char_accuracy += (pred_seq[mask] == true_seq[mask]).float().mean().item()
                
                total_samples += 1
        
        avg_loss = total_loss / total_tokens
        exact_match_accuracy = exact_matches / total_samples
        char_level_accuracy = char_accuracy / total_samples
        perplexity = math.exp(avg_loss)
        
        return avg_loss, exact_match_accuracy, char_level_accuracy, perplexity
    
    def train(self, num_epochs: int, checkpoint_dir: str = 'checkpoints'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_loss = self.train_epoch()
            print(f"Training Loss: {train_loss:.4f}")
            
            # Evaluation
            val_loss, exact_match, char_acc, perplexity = self.evaluate()
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Exact Match Accuracy: {exact_match:.4f}")
            print(f"Character Level Accuracy: {char_acc:.4f}")
            print(f"Perplexity: {perplexity:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'exact_match': exact_match,
                    'char_acc': char_acc,
                    'perplexity': perplexity
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
    
    @staticmethod
    def load_checkpoint(model: nn.Module, checkpoint_path: str, device: str = 'cuda') -> 'Trainer':
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        return checkpoint

def greedy_decode(model: nn.Module, src: torch.Tensor, src_vocab: Dict[str, int], 
                 tgt_vocab: Dict[str, int], max_len: int = 25, device: str = 'cuda') -> List[int]:
    model.eval()
    src = src.to(device)
    
    # Create source mask - [batch_size, 1, 1, src_len]
    src_mask = (src != src_vocab['<pad>']).unsqueeze(1).unsqueeze(2).to(device)
    
    # Create encoder output by running through encoder layers manually
    src_emb = model.src_embedding(src) * math.sqrt(model.d_model)
    src_emb = model.pos_encoding(src_emb)
    enc_output = model.dropout(src_emb)
    
    for enc_layer in model.encoder:
        enc_output = enc_layer(enc_output, src_mask)
    
    # Start with SOS token
    ys = torch.ones(1, 1).fill_(tgt_vocab['<sos>']).type_as(src.data).long().to(device)
    
    for i in range(max_len - 1):
        # Create target mask for sequence - [batch_size, 1, tgt_len, tgt_len]
        seq_len = ys.size(1)
        # Create a causal mask (lower triangular)
        causal_mask = torch.triu(torch.ones((1, seq_len, seq_len), device=device), diagonal=1) == 0
        # Add padding mask if needed (not needed for generated tokens as all are valid)
        tgt_mask = causal_mask
        
        # Get target embeddings
        tgt_emb = model.tgt_embedding(ys) 
        tgt_emb = model.pos_encoding(tgt_emb)
        tgt_emb = model.dropout(tgt_emb)
        
        # Run through decoder layers
        dec_output = tgt_emb
        for dec_layer in model.decoder:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)
        
        # Get prediction from final position
        out = model.fc(dec_output[:, -1])
        
        # Get next token
        _, next_word = torch.max(out, dim=1)
        next_word = next_word.item()
        
        # Add to output sequence
        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).long().fill_(next_word).to(device)], dim=1)
        
        # Stop if we predict EOS
        if next_word == tgt_vocab['<eos>']:
            break
    
    return ys.squeeze().tolist()

def create_vocabularies() -> Tuple[Dict[str, int], Dict[str, int]]:
    """Create source and target vocabularies for arithmetic problems."""
    # Special tokens
    special_tokens = ['<pad>', '<sos>', '<eos>']
    
    # Source vocabulary (arithmetic expressions)
    src_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-']
    src_vocab = {token: i for i, token in enumerate(special_tokens + src_chars)}
    
    # Target vocabulary (numbers and operators)
    tgt_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-']
    tgt_vocab = {token: i for i, token in enumerate(special_tokens + tgt_chars)}
    
    return src_vocab, tgt_vocab

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create vocabularies
    src_vocab, tgt_vocab = create_vocabularies()
    
    # Create datasets
    train_dataset = ArithmeticDataset('/kaggle/working/data/train.csv', src_vocab, tgt_vocab)
    val_dataset = ArithmeticDataset('/kaggle/working/data/val.csv', src_vocab, tgt_vocab)
    
    # Create collate function
    pad_collate_fn = PadCollate(src_vocab['<pad>'], tgt_vocab['<pad>'])
    
    # Create data loaders with optimized batch sizes
    train_loader = DataLoader(
        train_dataset,
        batch_size=256,  # Larger batch size for better GPU utilization
        shuffle=True,
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster data transfer to GPU
        collate_fn=pad_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,  # Larger batch size for validation
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=pad_collate_fn
    )
    
    # Model hyperparameters (optimized for arithmetic tasks)
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256,      
        num_heads=8,
        num_layers=3,      
        d_ff=1024,         
        dropout=0.1,
        max_len=25         # Maximum length for arithmetic problems
    )
    
    # Move model to device
    model = model.to(device)
    
    # Initialize trainer with optimized learning rate
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=0.00001,  # Slightly lower learning rate
        device=device
    )
    
    # Training parameters
    num_epochs = 50  # Reduced epochs with early stopping
    checkpoint_dir = 'model'  # Changed to model directory
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Training loop with early stopping
    best_val_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training
        train_loss = trainer.train_epoch()
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluation
        val_loss, exact_match, char_acc, perplexity = trainer.evaluate()
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Exact Match Accuracy: {exact_match:.4f}")
        print(f"Character Level Accuracy: {char_acc:.4f}")
        print(f"Perplexity: {perplexity:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': trainer.optimizer.state_dict(),
                'val_loss': val_loss,
                'exact_match': exact_match,
                'char_acc': char_acc,
                'perplexity': perplexity,
                'src_vocab': src_vocab,
                'tgt_vocab': tgt_vocab
            }, os.path.join(checkpoint_dir, 'best_model.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model for final evaluation
    checkpoint = torch.load(os.path.join(checkpoint_dir, 'best_model.pt'), map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Final evaluation
    print("\nFinal Evaluation:")
    val_loss, exact_match, char_acc, perplexity = trainer.evaluate()
    print(f"Final Validation Loss: {val_loss:.4f}")
    print(f"Final Exact Match Accuracy: {exact_match:.4f}")
    print(f"Final Character Level Accuracy: {char_acc:.4f}")
    print(f"Final Perplexity: {perplexity:.4f}")

if __name__ == "__main__":
    main() 