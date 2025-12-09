# file 2: data_loader.py
"""Data loading for CLINC150."""

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tqdm import tqdm


def load_clinc150():
    print("Loading CLINC150 from HuggingFace...")
    dataset = load_dataset("clinc_oos", "plus")
    intent_names = dataset['train'].features['intent'].names
    oos_id = intent_names.index('oos')

    # Create mapping: original intent ID → remapped ID (0-149)
    intent_to_label = {}
    label_count = 0
    for i, name in enumerate(intent_names):
        if i != oos_id:  # Skip OOS
            intent_to_label[i] = label_count
            label_count += 1

    print(f"Loaded: {len(intent_names)} classes")
    print(f"In-domain: {label_count} classes (remapped to 0-{label_count-1})")
    print(f"OOS ID: {oos_id}")

    return dataset, oos_id, intent_to_label


class CLINC150Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length, oos_id, intent_to_label):
        self.encodings = []
        self.labels = []
        self.is_oos = []
        self.texts = []

        print(f"Tokenizing {len(data['text'])} samples...")
        for text, intent in tqdm(zip(data['text'], data['intent']),
                                total=len(data['text']), leave=False):
            encoding = tokenizer(text, truncation=True, padding='max_length',
                               max_length=max_length, return_tensors='pt')

            self.encodings.append({
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze()
            })

            # Remap intent IDs
            if intent == oos_id:
                self.labels.append(-1)
                self.is_oos.append(True)
            else:
                self.labels.append(intent_to_label[intent])  # Remapped
                self.is_oos.append(False)

            self.texts.append(text)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings[idx]['input_ids'],
            'attention_mask': self.encodings[idx]['attention_mask'],
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'is_oos': torch.tensor(self.is_oos[idx], dtype=torch.bool),
            'text': self.texts[idx]
        }

    def __len__(self):
        return len(self.encodings)


def create_dataloaders(dataset, tokenizer, config, oos_id, intent_to_label):
    train_dataset = CLINC150Dataset(dataset['train'], tokenizer,
                                    config.max_seq_length, oos_id, intent_to_label)
    val_dataset = CLINC150Dataset(dataset['validation'], tokenizer,
                                  config.max_seq_length, oos_id, intent_to_label)
    test_dataset = CLINC150Dataset(dataset['test'], tokenizer,
                                   config.max_seq_length, oos_id, intent_to_label)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size,
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size,
                           shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size,
                            shuffle=False, num_workers=0)

    print(f"\nDataLoaders: {len(train_loader)} train, {len(val_loader)} val, {len(test_loader)} test")
    return train_loader, val_loader, test_loader