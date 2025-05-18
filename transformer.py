import math
from typing import Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(x.device)

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
        key_padding_mask = (~mask.squeeze(1).squeeze(1)) if mask is not None else None
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        return self.norm2(x + self.dropout2(ff_output))

class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        src_kpm = (~src_mask.squeeze(1).squeeze(1)) if src_mask is not None else None

        if tgt_mask is not None:
            seq_len = x.size(1)
            causal = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            tgt_kpm = (~tgt_mask[:, 0, :].bool()) if tgt_mask.dim()==3 else None
        else:
            causal, tgt_kpm = None, None

        # self-attn
        sa, _ = self.self_attn(x, x, x, attn_mask=causal, key_padding_mask=tgt_kpm)
        x = self.norm1(x + self.dropout1(sa))
        # cross-attn
        ca, _ = self.cross_attn(x, enc_output, enc_output, key_padding_mask=src_kpm)
        x = self.norm2(x + self.dropout2(ca))
        # feed-forward
        ff = self.feed_forward(x)
        return self.norm3(x + self.dropout3(ff))

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 25,
    ):
        super().__init__()
        self.d_model = d_model
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoding  = PositionalEncoding(d_model, max_len)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.fc      = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None
    ) -> torch.Tensor:
        src_emb = self.dropout(self.pos_encoding(self.src_embedding(src) * math.sqrt(self.d_model)))
        tgt_emb = self.dropout(self.pos_encoding(self.tgt_embedding(tgt)))
        enc = src_emb
        for layer in self.encoder:
            enc = layer(enc, src_mask)
        dec = tgt_emb
        for layer in self.decoder:
            dec = layer(dec, enc, src_mask, tgt_mask)
        return self.fc(dec)

def create_masks(
    src: torch.Tensor,
    tgt: torch.Tensor,
    pad_token: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = src.device
    src_mask = (src != pad_token).unsqueeze(1).unsqueeze(2)
    tgt_len, bs = tgt.size(1), tgt.size(0)
    causal = torch.triu(torch.ones((tgt_len, tgt_len), device=device), diagonal=1) == 0
    causal = causal.unsqueeze(0).expand(bs, -1, -1)
    pad_mask = (tgt != pad_token)
    pad_attn = pad_mask.unsqueeze(1).expand(-1, tgt_len, -1)
    tgt_mask = causal & pad_attn
    return src_mask, tgt_mask

class PadCollate:
    def __init__(self, src_pad_idx: int, tgt_pad_idx: int):
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

    def __call__(self, batch):
        src_batch, tgt_batch = zip(*batch)
        max_s = max(t.size(0) for t in src_batch)
        max_t = max(t.size(0) for t in tgt_batch)
        padded_src = torch.full((len(batch), max_s), self.src_pad_idx, dtype=torch.long)
        padded_tgt = torch.full((len(batch), max_t), self.tgt_pad_idx, dtype=torch.long)
        for i, (s, t) in enumerate(batch):
            padded_src[i, :s.size(0)] = s
            padded_tgt[i, :t.size(0)] = t
        return padded_src, padded_tgt

def greedy_decode(
    model: nn.Module,
    src: torch.Tensor,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    max_len: int = 25,
    device: str = 'cuda',
) -> List[int]:
    model.eval()
    src = src.to(device)
    src_mask = (src != src_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
    emb = model.src_embedding(src) * math.sqrt(model.d_model)
    emb = model.pos_encoding(emb)
    enc = model.dropout(emb)
    for layer in model.encoder:
        enc = layer(enc, src_mask)
    ys = torch.ones(1,1).fill_(tgt_vocab['<sos>']).type_as(src).long().to(device)
    for _ in range(max_len-1):
        seq_len = ys.size(1)
        causal = torch.triu(torch.ones((1,seq_len,seq_len),device=device), diagonal=1)==0
        emb_t = model.tgt_embedding(ys)
        emb_t = model.pos_encoding(emb_t)
        emb_t = model.dropout(emb_t)
        dec = emb_t
        for layer in model.decoder:
            dec = layer(dec, enc, src_mask, causal)
        out = model.fc(dec[:,-1])
        _, nxt = out.max(dim=1)
        nxt = nxt.item()
        ys = torch.cat([ys, nxt*torch.ones(1,1,device=device,dtype=ys.dtype)], dim=1)
        if nxt == tgt_vocab['<eos>']:
            break
    return ys.squeeze().tolist()
