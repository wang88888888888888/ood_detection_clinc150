# file 8: main.py
"""Main pipeline."""

import torch
import numpy as np
from transformers import DistilBertTokenizer

from config import Config
from data_loader import load_clinc150, create_dataloaders
from model import IntentClassifier
from trainer import Trainer
from evaluator import Evaluator
from utils import inspect_data_samples, analyze_mc_dropout, plot_learning_curves
from visualization import ResultsVisualizer

def main():
    config = Config()
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    print("CLINC150 OOD Detection: M1 vs M2\n" + "="*60)

    # Data
    dataset, oos_id, intent_to_label = load_clinc150()
    tokenizer = DistilBertTokenizer.from_pretrained(config.model_name)
    inspect_data_samples(dataset, tokenizer, oos_id, n=10)

    train_loader, val_loader, test_loader = create_dataloaders(
        dataset, tokenizer, config, oos_id, intent_to_label
    )

    # Train
    print("\n" + "="*60 + "\nTRAINING")
    model = IntentClassifier(config.num_classes, config.dropout_rate)
    print(f"Model: {config.num_classes} classes, dropout={config.dropout_rate}")
    trainer = Trainer(model, config.device, config)
    best_val = trainer.train_model(train_loader, val_loader, config)
    print(f"\nBest validation: {best_val:.4f}")
    plot_learning_curves(trainer.history)

    # MC inspection
    batch = next(iter(test_loader))
    in_idx = (~batch['is_oos']).nonzero(as_tuple=True)[0][0]
    analyze_mc_dropout(
        model,
        batch['input_ids'][in_idx:in_idx+1].to(config.device),
        batch['attention_mask'][in_idx:in_idx+1].to(config.device),
        batch['label'][in_idx].item(),
        batch['text'][in_idx]
    )

    if batch['is_oos'].any():
        oos_idx = batch['is_oos'].nonzero(as_tuple=True)[0][0]
        analyze_mc_dropout(
            model,
            batch['input_ids'][oos_idx:oos_idx+1].to(config.device),
            batch['attention_mask'][oos_idx:oos_idx+1].to(config.device),
            batch['label'][oos_idx].item(),
            batch['text'][oos_idx]
        )

    # Evaluate
    print("\n" + "="*60 + "\nEVALUATION")
    evaluator = Evaluator()
    raw_results, metrics = evaluator.evaluate(model, test_loader, config.device, config.num_mc_samples)

    if 'intent_accuracy' in metrics:
        print(f"\nIntent Classification:")
        print(f"  M1: {metrics['intent_accuracy']['m1']:.1%}")
        print(f"  M2: {metrics['intent_accuracy']['m2']:.1%}")

    if 'ood_detection' in metrics:
        print(f"\nOOD Detection:")
        for method, scores in metrics['ood_detection'].items():
            print(f"  {method}: AUROC={scores['auroc']:.1%}, AUPR={scores['aupr']:.1%}")

    # Visualization section
    print("\n" + "="*60 + "\nVISUALIZATION")
    print("🔄 Processing detailed results for visualization...")
    detailed_results = evaluator.process_detailed_results(raw_results)

    print(f" Processed {len(detailed_results)} samples for visualization")

    visualizer = ResultsVisualizer(detailed_results)
    visualizer.generate_all_plots()

    return {'model': model, 'metrics': metrics, 'raw_results': raw_results}

if __name__ == "__main__":
    results = main()