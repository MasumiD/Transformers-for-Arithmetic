# model_analysis.py

import torch
import torch.nn as nn
import math
import os
import argparse
import csv
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List
import time
from tqdm import tqdm

from transformer import Transformer, create_masks, greedy_decode, PadCollate
from create_datasets import ArithmeticDataset

def decode_with_model(
    model: nn.Module,
    problem: str,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    device: str
) -> str:
    # map chars → token IDs, with <sos>/<eos>
    src_tokens = [src_vocab['<sos>']] + \
                 [src_vocab.get(c, src_vocab['<pad>']) for c in problem] + \
                 [src_vocab['<eos>']]

    src = torch.tensor([src_tokens], dtype=torch.long).to(device)

    # get the raw token list
    out_tokens = greedy_decode(
        model, src,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=model.pos_encoding.pe.size(1),  # or model.max_len if you store it
        device=device
    )

    # invert vocab
    inv_tgt = {i: t for t, i in tgt_vocab.items()}

    # build string
    s = ""
    for tok in out_tokens:
        if tok == tgt_vocab['<sos>'] or tok == tgt_vocab['<pad>']:
            continue
        if tok == tgt_vocab['<eos>']:
            break
        s += inv_tgt[tok]

    # strip spurious leading zeros (but keep “0” if that’s all we have)
    if len(s) > 1 and s[0] == '0':
        s = s.lstrip('0')
        if s == '':
            s = '0'

    return s

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    pad_token: int,
    device: str
) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = total_tokens = 0
    exact_matches = char_acc = total_samples = 0

    for src, tgt in tqdm(data_loader, desc='Evaluating'):
        src, tgt = src.to(device), tgt.to(device)
        src_mask, tgt_mask = create_masks(src, tgt, pad_token)

        out = model(src, tgt[:, :-1], src_mask, tgt_mask[:, :-1, :-1])
        loss = criterion(out.view(-1, out.size(-1)),
                         tgt[:, 1:].reshape(-1))

        non_pad = (tgt[:, 1:] != pad_token)
        total_loss  += loss.item() * non_pad.sum().item()
        total_tokens += non_pad.sum().item()

        preds = out.argmax(dim=-1)
        for i in range(len(preds)):
            true_seq = tgt[i,1:][non_pad[i]]
            pred_seq = preds[i][non_pad[i]]
            if torch.equal(pred_seq, true_seq):
                exact_matches += 1
            char_acc += (pred_seq == true_seq).float().mean().item()
            total_samples += 1

    avg_loss = total_loss / total_tokens
    return (
        avg_loss,
        exact_matches / total_samples,
        char_acc / total_samples,
        math.exp(avg_loss)
    )

@torch.no_grad()
def evaluate_and_save_predictions(
    model: nn.Module,
    dataset_path: str,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    device: str,
    output_csv: str
):
    ds = ArithmeticDataset(dataset_path, src_vocab, tgt_vocab)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['problem','expected','predicted','correct'])
        for i in tqdm(range(len(ds)), desc='Predicting'):
            prob, exp = ds.get_raw_item(i)
            pred = decode_with_model(model, prob, src_vocab, tgt_vocab, device)
            writer.writerow([prob, exp, pred, pred == exp])

    print(f"Saved predictions to {output_csv}")


def evaluate_on_dataset(
    model: nn.Module,
    dataset_path: str,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    device: str,
    batch_size: int = 128
) -> dict:
    ds = ArithmeticDataset(dataset_path, src_vocab, tgt_vocab)
    collate = PadCollate(src_vocab['<pad>'], tgt_vocab['<pad>'])
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True,
                        num_workers=2, collate_fn=collate)

    crit = nn.CrossEntropyLoss(ignore_index=tgt_vocab['<pad>'])
    start = time.time()
    loss, em, ca, ppl = evaluate_model(model, loader, crit, tgt_vocab['<pad>'], device)
    elapsed = time.time() - start

    print(f"Dataset: {dataset_path}")
    print(f"  Loss: {loss:.4f}")
    print(f"  Exact match: {em:.4f}")
    print(f"  Char acc: {ca:.4f}")
    print(f"  Perplexity: {ppl:.4f}")
    print(f"  Time: {elapsed:.2f}s\n" + "-"*40)

    return {"loss":loss, "exact_match":em, "char_acc":ca, "perplexity":ppl, "time":elapsed}

def test_sample_problems(
    model: nn.Module,
    src_vocab: Dict[str, int],
    tgt_vocab: Dict[str, int],
    device: str
):
    print("\nSample Problems\n" + "-"*40)
    for prob in ["123+456","789-321","25+38","100-99","1234+5678","9999-8888"]:
        out = decode_with_model(model, prob, src_vocab, tgt_vocab, device)
        exp = str(eval(prob))
        print(f"{prob} → model={out} vs expected={exp}  {'✅' if out==exp else '❌'}")

def get_model_params(version: str) -> dict:
    if version=="v3":
        return {"d_model":256,"num_heads":8,"num_layers":3,
                "d_ff":1024,"dropout":0.1,"max_len":25,
                "model_path":"V3/working/model/best_model.pt"}
    if version=="v4":
        return {"d_model":128,"num_heads":4,"num_layers":2,
                "d_ff":512,"dropout":0.1,"max_len":25,
                "model_path":"V4/working/model/best_model.pt"}
    if version=="v5":
        return {"d_model":256,"num_heads":8,"num_layers":4,
                "d_ff":2048,"dropout":0.1,"max_len":25,
                "model_path":"V5/working/model/best_model.pt"}
    raise ValueError(f"Unknown version {version}")

def main():
    p = argparse.ArgumentParser(description="Analyze arithmetic transformers")
    p.add_argument("--model", choices=["v3","v4","v5","all"], default="all")
    p.add_argument("--test_dataset", default="data/test.csv")
    p.add_argument("--gen_dataset",  default="data/generalization.csv")
    p.add_argument("--batch_size",   type=int, default=128)
    p.add_argument("--save_predictions", action="store_true")
    p.add_argument("--results_dir",  default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    to_analyze = ["v3","v4","v5"] if args.model=="all" else [args.model]
    all_results = {}

    for ver in to_analyze:
        print("\n" + "="*60)
        print(f"Model {ver.upper()}")
        print("="*60)
        params = get_model_params(ver)
        ckpt = torch.load(params.pop("model_path"), map_location=device)
        src_vocab, tgt_vocab = ckpt["src_vocab"], ckpt["tgt_vocab"]
        model = Transformer(len(src_vocab), len(tgt_vocab), **params).to(device)
        model.load_state_dict(ckpt["model_state_dict"])

        print(f"Loaded from {ckpt.get('epoch','?')} epochs")
        test_res = evaluate_on_dataset(model, args.test_dataset, src_vocab, tgt_vocab, device, args.batch_size)
        gen_res  = evaluate_on_dataset(model, args.gen_dataset,  src_vocab, tgt_vocab, device, args.batch_size)
        test_sample_problems(model, src_vocab, tgt_vocab, device)

        if args.save_predictions:
            os.makedirs(args.results_dir, exist_ok=True)
            evaluate_and_save_predictions(
                model, args.test_dataset, src_vocab, tgt_vocab, device,
                os.path.join(args.results_dir, f"{ver}_test_preds.csv")
            )
            evaluate_and_save_predictions(
                model, args.gen_dataset, src_vocab, tgt_vocab, device,
                os.path.join(args.results_dir, f"{ver}_gen_preds.csv")
            )

        all_results[ver] = {"test":test_res, "gen":gen_res}

    if len(all_results)>1:
        print("\nComparison on Test Set".center(60,"-"))
        print(f"{'ver':<5} {'loss':<8} {'exact':<8} {'char':<8} {'ppl':<8}")
        for ver, res in all_results.items():
            r = res["test"]
            print(f"{ver:<5} {r['loss']:<8.4f} {r['exact_match']:<8.4f}"
                  f" {r['char_acc']:<8.4f} {r['perplexity']:<8.4f}")

        print("\nBest by exact match:",
              max(all_results, key=lambda v: all_results[v]["test"]["exact_match"]))

if __name__=="__main__":
    main()
