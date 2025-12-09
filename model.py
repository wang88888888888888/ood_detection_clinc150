# file 3: model.py
"""DistilBERT intent classifier."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DistilBertModel


class IntentClassifier(nn.Module):
    """
    Intent classifier supporting:
    - forward(): Standard inference (dropout OFF)
    - monte_carlo_forward(): MC Dropout (dropout ON)
    """

    def __init__(self, num_classes=150, dropout_rate=0.1,
                 model_name='distilbert-base-uncased'):
        super().__init__()
        self.num_classes = num_classes

        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_classes)

        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0]
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)
        return logits

    def monte_carlo_forward(self, input_ids, attention_mask, num_samples=50):
        was_training = self.training
        self.train()

        with torch.no_grad():
            predictions = []
            for _ in range(num_samples):
                logits = self.forward(input_ids, attention_mask)
                probs = F.softmax(logits, dim=-1)
                predictions.append(probs)

            predictions = torch.stack(predictions)
            mean_predictions = torch.mean(predictions, dim=0)
            predictive_variance = torch.var(predictions, dim=0).mean(dim=-1)
            predictive_entropy = -torch.sum(
                mean_predictions * torch.log(mean_predictions + 1e-8), dim=-1
            )

        self.train(was_training)

        return {
            'mean_predictions': mean_predictions,
            'predictive_variance': predictive_variance,
            'predictive_entropy': predictive_entropy,
            'all_predictions': predictions,
        }