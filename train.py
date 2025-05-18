import os
import math
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from create_datasets import create_datasets, save_dataset
from create_datasets import ArithmeticDataset, create_vocabularies
from transformer import Transformer, create_masks, PadCollate

class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        learning_rate: float = 1e-5,
        device: str = 'cuda'
    ):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = torch.nn.CrossEntropyLoss(ignore_index=train_loader.dataset.tgt_vocab['<pad>'])
        self.optimizer    = optim.Adam(self.model.parameters(), lr=learning_rate, betas=(0.9,0.98), eps=1e-9)
        self.scheduler    = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2)
        self.best_val_loss = float('inf')

    def train_epoch(self) -> float:
        self.model.train()
        total_loss, total_tokens = 0.0, 0
        for src, tgt in tqdm(self.train_loader, desc='Training'):
            src, tgt = src.to(self.device), tgt.to(self.device)
            src_mask, tgt_mask = create_masks(src, tgt, self.train_loader.dataset.src_vocab['<pad>'])
            self.optimizer.zero_grad()
            out = self.model(src, tgt[:,:-1], src_mask, tgt_mask[:,:-1,:-1])
            loss = self.criterion(out.reshape(-1,out.size(-1)), tgt[:,1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            non_pad = (tgt[:,1:] != self.train_loader.dataset.tgt_vocab['<pad>'])
            total_loss  += loss.item() * non_pad.sum().item()
            total_tokens += non_pad.sum().item()
        return total_loss / total_tokens

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, total_tokens = 0.0, 0
        exact_matches, char_acc, samples = 0, 0.0, 0
        for src, tgt in tqdm(self.val_loader, desc='Evaluating'):
            src, tgt = src.to(self.device), tgt.to(self.device)
            src_mask, tgt_mask = create_masks(src, tgt, self.val_loader.dataset.src_vocab['<pad>'])
            out = self.model(src, tgt[:,:-1], src_mask, tgt_mask[:,:-1,:-1])
            loss = self.criterion(out.reshape(-1,out.size(-1)), tgt[:,1:].reshape(-1))
            non_pad = (tgt[:,1:] != self.val_loader.dataset.tgt_vocab['<pad>'])
            total_loss  += loss.item() * non_pad.sum().item()
            total_tokens += non_pad.sum().item()

            preds = out.argmax(dim=-1)
            for i in range(len(preds)):
                pred_seq, true_seq = preds[i], tgt[i,1:]
                if torch.equal(pred_seq, true_seq):
                    exact_matches += 1
                mask = true_seq != self.val_loader.dataset.tgt_vocab['<pad>']
                if mask.any():
                    char_acc += (pred_seq[mask] == true_seq[mask]).float().mean().item()
                samples += 1

        avg_loss = total_loss / total_tokens
        return (
            avg_loss,
            exact_matches / samples,
            char_acc / samples,
            math.exp(avg_loss)
        )

    def train(self, num_epochs: int, checkpoint_dir: str = 'model'):
        os.makedirs(checkpoint_dir, exist_ok=True)
        for epoch in range(1, num_epochs+1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            tr_loss = self.train_epoch()
            print(f" Training Loss: {tr_loss:.4f}")
            val_loss, ex_acc, ch_acc, perp = self.evaluate()
            print(f" Validation Loss: {val_loss:.4f}")
            print(f" Exact Match: {ex_acc:.4f}, Char Acc: {ch_acc:.4f}, Perplexity: {perp:.4f}")
            self.scheduler.step(val_loss)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss
                }, os.path.join(checkpoint_dir, 'best_model.pt'))

def main():
    # 1) build & save datasets
    os.makedirs('data', exist_ok=True)
    ds = create_datasets()
    save_dataset(ds['train'], 'data/train.csv')
    save_dataset(ds['val'],   'data/val.csv')
    save_dataset(ds['test'],  'data/test.csv')
    save_dataset(ds['generalization'], 'data/generalization.csv')
    print("Datasets written to data/")

    # 2) prepare dataloaders
    src_vocab, tgt_vocab = create_vocabularies()
    train_ds = ArithmeticDataset('data/train.csv', src_vocab, tgt_vocab)
    val_ds   = ArithmeticDataset('data/val.csv',   src_vocab, tgt_vocab)
    collate  = PadCollate(src_vocab['<pad>'], tgt_vocab['<pad>'])
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True,  num_workers=4, pin_memory=True, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=256, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate)

    # 3) model & trainer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=256, num_heads=8, num_layers=3, d_ff=1024, dropout=0.1, max_len=25
    )
    trainer = Trainer(model, train_loader, val_loader, learning_rate=1e-5, device=device)

    # 4) training
    trainer.train(num_epochs=50, checkpoint_dir='model')
    print("\nTraining completed!  Best val loss:", trainer.best_val_loss)

    # 5) final evaluation
    ckpt = torch.load(os.path.join('model','best_model.pt'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    val_loss, ex_acc, ch_acc, perp = trainer.evaluate()
    print(f"\nFinal → Loss: {val_loss:.4f} | Exact: {ex_acc:.4f} | Char: {ch_acc:.4f} | Ppl: {perp:.4f}")

if __name__ == "__main__":
    main()
