import torch
import os
import csv
from tqdm import tqdm
from transformer import Transformer
from model_analysis import decode_with_model

def regenerate_prediction_csv(model_version, output_csv, dataset_path):
    """Regenerate predictions CSV file for a given model version."""
    print(f"Regenerating predictions for {model_version}...")
    
    # Define model parameters based on version
    if model_version == "v3":
        model_params = {
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 3,
            "d_ff": 1024,
            "dropout": 0.1,
            "max_len": 25,
            "model_path": "V3/working/model/best_model.pt"
        }
    elif model_version == "v4":
        model_params = {
            "d_model": 128,
            "num_heads": 4,
            "num_layers": 2,
            "d_ff": 512,
            "dropout": 0.1,
            "max_len": 25,
            "model_path": "V4/working/model/best_model.pt"
        }
    elif model_version == "v5":
        model_params = {
            "d_model": 256,
            "num_heads": 8,
            "num_layers": 4,
            "d_ff": 2048,
            "dropout": 0.1,
            "max_len": 25,
            "model_path": "V5/working/model/best_model.pt"
        }
    else:
        raise ValueError(f"Unknown model version: {model_version}")
    
    # Extract model path
    model_path = model_params.pop("model_path")
    
    # Load checkpoint
    device = torch.device('cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get vocabularies
    src_vocab = checkpoint['src_vocab']
    tgt_vocab = checkpoint['tgt_vocab']
    
    # Create model
    model = Transformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        **model_params
    )
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Read dataset
    with open(dataset_path, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
        data = [line.strip().split(',') for line in lines]
    
    # Create directory for output if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Write predictions to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['problem', 'expected', 'predicted', 'correct'])
        
        correct_count = 0
        total_count = 0
        
        # Process each item
        for problem, expected in tqdm(data, desc=f'Generating {model_version} predictions'):
            # Predict using the model
            predicted = decode_with_model(model, problem, src_vocab, tgt_vocab, device)
            
            # Check if prediction is correct
            correct = predicted == expected
            if correct:
                correct_count += 1
            total_count += 1
            
            # Write to CSV
            writer.writerow([problem, expected, predicted, correct])
    
    accuracy = correct_count / total_count if total_count > 0 else 0
    print(f"Predictions saved to {output_csv}")
    print(f"Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    return accuracy

def main():
    # Datasets
    test_dataset = 'data/test.csv'
    gen_dataset = 'data/generalization.csv'
    
    # Regenerate v4 predictions
    v4_test_accuracy = regenerate_prediction_csv("v4", "results/v4_test_predictions_corrected.csv", test_dataset)
    v4_gen_accuracy = regenerate_prediction_csv("v4", "results/v4_gen_predictions_corrected.csv", gen_dataset)
    
    # Regenerate v5 predictions
    v5_test_accuracy = regenerate_prediction_csv("v5", "results/v5_test_predictions_corrected.csv", test_dataset)
    v5_gen_accuracy = regenerate_prediction_csv("v5", "results/v5_gen_predictions_corrected.csv", gen_dataset)
    
    # Print summary
    print("\nSummary of Results:")
    print("-" * 50)
    print(f"V4 Test Accuracy: {v4_test_accuracy:.4f}")
    print(f"V4 Generalization Accuracy: {v4_gen_accuracy:.4f}")
    print(f"V5 Test Accuracy: {v5_test_accuracy:.4f}")
    print(f"V5 Generalization Accuracy: {v5_gen_accuracy:.4f}")

if __name__ == "__main__":
    main() 