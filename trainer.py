# file 4: trainer.py
"""Training logic."""

import torch
import torch.nn as nn
from tqdm import tqdm


def filter_in_domain(batch):
    mask = ~batch['is_oos']
    if mask.sum() == 0:
        return None
    return {
        'input_ids': batch['input_ids'][mask],
        'attention_mask': batch['attention_mask'][mask],
        'labels': batch['label'][mask]
    }


class Trainer:
    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(train_loader, desc="Training", leave=False):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            is_oos = batch['is_oos'].to(self.device)

            filtered = filter_in_domain({
                'input_ids': input_ids, 'attention_mask': attention_mask,
                'label': labels, 'is_oos': is_oos
            })
            if filtered is None:
                continue

            self.optimizer.zero_grad()
            logits = self.model(filtered['input_ids'], filtered['attention_mask'])
            loss = self.criterion(logits, filtered['labels'])
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(1) == filtered['labels']).sum().item()
            total += filtered['labels'].size(0)

        return total_loss / len(train_loader), correct / total if total > 0 else 0

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating", leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                is_oos = batch['is_oos'].to(self.device)

                filtered = filter_in_domain({
                    'input_ids': input_ids, 'attention_mask': attention_mask,
                    'label': labels, 'is_oos': is_oos
                })
                if filtered is None:
                    continue

                logits = self.model(filtered['input_ids'], filtered['attention_mask'])
                loss = self.criterion(logits, filtered['labels'])

                total_loss += loss.item()
                correct += (logits.argmax(1) == filtered['labels']).sum().item()
                total += filtered['labels'].size(0)

        return total_loss / len(val_loader), correct / total if total > 0 else 0

    def train_model(self, train_loader, val_loader, config):
        best_val_acc = 0
        patience_counter = 0
        best_state = None

        for epoch in range(config.num_epochs):
            print(f"\nEpoch {epoch+1}/{config.num_epochs}")

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            print(f"Train: Loss={train_loss:.4f}, Acc={train_acc:.4f}")
            print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc:.4f}")

            if val_acc - best_val_acc > config.min_improvement:
                best_val_acc = val_acc
                best_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"✓ New best: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print("Early stopping")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        return best_val_acc