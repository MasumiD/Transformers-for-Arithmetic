import random
import os
import csv
from typing import Tuple, List, Dict
import os
from torch.utils.data import Dataset
import torch

def generate_arithmetic_sample(min_digits: int, max_digits: int, operators: List[str] = ["+", "-"]):
    """
    Generate a random arithmetic sample (addition or subtraction).
    """
    op = random.choice(operators)
    
    def random_number(n_digits: int) -> int:
        if n_digits == 1:
            return random.randint(0, 9)
        lower = 10 ** (n_digits - 1)
        upper = 10 ** n_digits - 1
        return random.randint(lower, upper)
    
    len_a = random.randint(min_digits, max_digits)
    len_b = random.randint(min_digits, max_digits)
    
    a = random_number(len_a)
    b = random_number(len_b)
    
    problem = f"{a}{op}{b}"
    if op == "+":
        answer = a + b
    else:
        answer = a - b
    return problem, str(answer)

# Edge-case generators
def generate_addition_carry_example(n_digits: int) -> Tuple[str, str]:
    """
    Generate one addition example with at least a carry in the unit place.
    """
    lower = 10 ** (n_digits - 1)
    upper = 10 ** n_digits - 1
    while True:
        a = random.randint(lower, upper)
        b = random.randint(lower, upper)
        if (a % 10) + (b % 10) >= 10:
            return f"{a}+{b}", str(a + b)

def generate_subtraction_borrow_example(n_digits: int) -> Tuple[str, str]:
    """
    Generate one subtraction example requiring a borrow in the unit place.
    """
    lower = 10 ** (n_digits - 1)
    upper = 10 ** n_digits - 1
    while True:
        a = random.randint(lower, upper)
        b = random.randint(lower, upper)
        if a > b and (a % 10) < (b % 10):
            return f"{a}-{b}", str(a - b)

# Dataset creation
def generate_dataset(num_examples: int, min_digits: int, max_digits: int,
                    operators: List[str] = ["+", "-"]):
    dataset = []
    for _ in range(num_examples):
        dataset.append(generate_arithmetic_sample(min_digits, max_digits, operators))
    return dataset

def create_datasets() -> Dict[str, List[Tuple[str, str]]]:
    # Baseline sizes
    train_size, val_size, test_size = 200000, 2000, 2000
    gen_size = 2000

    # In-distribution (1–3 digits)
    train_data = generate_dataset(train_size, 1, 3)
    val_data   = generate_dataset(val_size,   1, 3)
    test_data  = generate_dataset(test_size,  1, 3)

    # Generalization (4–6 digits)
    generalization_data = generate_dataset(gen_size, 1, 6)

    # Inject edge cases into training set
    edge_cases: List[Tuple[str, str]] = []
    for _ in range(100000):
        edge_cases.append(generate_addition_carry_example(3))
    for _ in range(100000):
        edge_cases.append(generate_subtraction_borrow_example(3))
    random.shuffle(edge_cases)
    train_data.extend(edge_cases)
    random.shuffle(train_data)

    return {
        "train": train_data,
        "val":   val_data,
        "test":  test_data,
        "generalization": generalization_data
    }

# Utility: Save to CSV
def save_dataset(dataset: List[Tuple[str, str]], filename: str):
    os.makedirs("data", exist_ok=True)
    path = os.path.join("data", filename)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["problem", "solution"]);
        for problem, solution in dataset:
            writer.writerow([problem, solution])

class ArithmeticDataset(Dataset):
    def __init__(self, data_path: str, src_vocab: Dict[str, int], tgt_vocab: Dict[str, int]):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.pad_token = '<pad>'
        self.sos_token = '<sos>'
        self.eos_token = '<eos>'

        with open(data_path, 'r') as f:
            lines = f.readlines()[1:]  # skip header
            self.data = [line.strip().split(',') for line in lines]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        problem, solution = self.data[idx]
        src_tokens = [self.src_vocab[self.sos_token]] + \
                     [self.src_vocab.get(c, self.src_vocab[self.pad_token]) for c in problem] + \
                     [self.src_vocab[self.eos_token]]
        tgt_tokens = [self.tgt_vocab[self.sos_token]] + \
                     [self.tgt_vocab.get(c, self.tgt_vocab[self.pad_token]) for c in solution] + \
                     [self.tgt_vocab[self.eos_token]]
        return torch.tensor(src_tokens, dtype=torch.long), torch.tensor(tgt_tokens, dtype=torch.long)

def create_vocabularies() -> Tuple[Dict[str, int], Dict[str, int]]:
    special_tokens = ['<pad>', '<sos>', '<eos>']
    src_chars = [str(i) for i in range(10)] + ['+', '-']
    tgt_chars = [str(i) for i in range(10)] + ['-']
    src_vocab = {tok: i for i, tok in enumerate(special_tokens + src_chars)}
    tgt_vocab = {tok: i for i, tok in enumerate(special_tokens + tgt_chars)}
    return src_vocab, tgt_vocab

if __name__ == "__main__":
    datasets = create_datasets()

    # Save each split
    save_dataset(datasets["train"], "train.csv")
    save_dataset(datasets["val"], "val.csv")
    save_dataset(datasets["test"], "test.csv")
    save_dataset(datasets["generalization"], "generalization.csv")

    print("Datasets written to data/train.csv, val.csv, test.csv and generalization.csv")
