# file 6: utils.py
"""Inspection and visualization utilities."""

import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt


def inspect_data_samples(dataset, tokenizer, oos_id, n=10):
    print("\n" + "="*60)
    print("DATA SAMPLES")
    print("="*60)

    for i in range(n):
        text = dataset['train']['text'][i]
        intent = dataset['train']['intent'][i]
        encoding = tokenizer(text, max_length=64, padding='max_length',
                           truncation=True, return_tensors='pt')
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0][:10])

        print(f"\n{i+1}. '{text}'")
        print(f"   Intent: {intent}, OOS: {intent==oos_id}, Label: {-1 if intent==oos_id else intent}")
        print(f"   Tokens: {tokens}")


def analyze_mc_dropout(model, input_ids, attention_mask, true_label, text):
    print("\n" + "="*60)
    print(f"MC DROPOUT: '{text}' (label={true_label})")
    print("="*60)

    model.train()
    predictions = []

    print(f"\n{'Run':<5} {'Top':<5} {'Conf':<8} Top 3")
    print("-"*60)

    with torch.no_grad():
        for i in range(50):
            logits = model(input_ids, attention_mask)
            probs = F.softmax(logits, dim=-1)[0]
            top = probs.argmax().item()
            predictions.append(probs.cpu().numpy())

            top3 = [f"{idx}:{probs[idx]:.2f}"
                   for idx in probs.argsort(descending=True)[:3]]
            print(f"{i+1:<5} {top:<5} {probs[top]:<8.4f} {', '.join(top3)}")

    predictions = np.array(predictions)
    mean = predictions.mean(0)
    final_pred = mean.argmax()
    std = predictions[:, final_pred].std()

    print(f"\nAggregated:")
    print(f"  Prediction: {final_pred}, Mean: {mean[final_pred]:.4f}, Std: {std:.4f}")
    print(f"  ±1σ: [{mean[final_pred]-std:.4f}, {mean[final_pred]+std:.4f}]")
    print(f"  Variance: {predictions.var(0).mean():.6f}, Entropy: {-(mean*np.log(mean+1e-8)).sum():.4f}")


def plot_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history['train_loss']) + 1)

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val')
    ax1.set(xlabel='Epoch', ylabel='Loss', title='Loss')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train')
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val')
    ax2.set(xlabel='Epoch', ylabel='Accuracy', title='Accuracy')
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=150)
    print("Saved learning_curves.png")
    plt.show()