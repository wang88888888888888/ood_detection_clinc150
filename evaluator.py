# file 5: evaluator.py
"""Evaluation for M1 vs M2 with complete metrics."""

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc


class Evaluator:
    def evaluate(self, model, test_loader, device, num_mc_samples):
        model.eval()
        results = {
            'true_labels': [], 'is_oos': [],
            # M1 metrics
            'm1_predictions': [], 'm1_probabilities': [], 'm1_entropy': [],
            # M2 metrics
            'm2_predictions': [], 'm2_probabilities': [], 'm2_max_conf': [],
            'mc_variance': [], 'mc_entropy': [],
            # New: save all MC sampling results
            'm2_all_predictions': []
        }

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating", leave=False):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # M1: Single-pass inference
                m1_logits = model(input_ids, attention_mask)
                m1_probs = F.softmax(m1_logits, dim=-1).cpu().numpy()
                m1_entropy = -(m1_probs * np.log(m1_probs + 1e-8)).sum(axis=1)

                # M2: MC Dropout inference
                mc_results = model.monte_carlo_forward(input_ids, attention_mask,
                                                      num_samples=num_mc_samples)
                m2_probs = mc_results['mean_predictions'].cpu().numpy()
                m2_max_conf = np.max(m2_probs, axis=1)

                # Store all results
                results['true_labels'].extend(batch['label'].cpu().numpy())
                results['is_oos'].extend(batch['is_oos'].cpu().numpy())
                results['m1_predictions'].extend(np.argmax(m1_probs, axis=1))
                results['m1_probabilities'].extend(m1_probs)
                results['m1_entropy'].extend(m1_entropy)
                results['m2_predictions'].extend(np.argmax(m2_probs, axis=1))
                results['m2_probabilities'].extend(m2_probs)
                results['m2_max_conf'].extend(m2_max_conf)
                results['mc_variance'].extend(mc_results['predictive_variance'].cpu().numpy())
                results['mc_entropy'].extend(mc_results['predictive_entropy'].cpu().numpy())

                # Save all 50 MC sampling results - per sample
                batch_mc_preds = mc_results['all_predictions'].cpu().numpy()  # [50, batch_size, 150]
                for i in range(batch_mc_preds.shape[1]):  # Iterate through each sample in the batch
                    results['m2_all_predictions'].append(batch_mc_preds[:, i, :])  # [50, 150]

        # Convert to numpy arrays, except for m2_all_predictions which remains a list
        raw_results = {}
        for k, v in results.items():
            if k == 'm2_all_predictions':
                raw_results[k] = v  # Keep as list of arrays
            else:
                raw_results[k] = np.array(v)
        metrics = self._compute_metrics(raw_results)

        return raw_results, metrics

    def _compute_metrics(self, results):
        is_oos = results['is_oos'].astype(bool)
        in_domain_mask = ~is_oos
        metrics = {}

        # Intent accuracy
        if in_domain_mask.sum() > 0:
            in_domain_true = results['true_labels'][in_domain_mask]
            m1_acc = accuracy_score(in_domain_true, results['m1_predictions'][in_domain_mask])
            m2_acc = accuracy_score(in_domain_true, results['m2_predictions'][in_domain_mask])
            metrics['intent_accuracy'] = {'m1': m1_acc, 'm2': m2_acc}

        # OOD detection - All 5 metrics
        if is_oos.sum() > 0:
            binary_labels = is_oos.astype(int)

            ood_scores = {
                'm1_max_confidence': 1 - np.max(results['m1_probabilities'], axis=1),
                'm1_entropy': results['m1_entropy'],
                'm2_max_confidence': 1 - results['m2_max_conf'],
                'm2_variance': results['mc_variance'],
                'm2_entropy': results['mc_entropy']
            }

            metrics['ood_detection'] = {}
            for name, scores in ood_scores.items():
                auroc = roc_auc_score(binary_labels, scores)
                precision, recall, _ = precision_recall_curve(binary_labels, scores)
                aupr = auc(recall, precision)
                metrics['ood_detection'][name] = {'auroc': auroc, 'aupr': aupr}

        self.print_summary(metrics)
        return metrics

    def process_detailed_results(self, results):
        """Process detailed results for visualization"""
        detailed_data = []

        for i in range(len(results['true_labels'])):
            # Get all 50 MC results for this sample
            mc_all_preds_sample = results['m2_all_predictions'][i]  # [50, 150]

            # Calculate confidence and entropy for each run
            mc_confidences = [np.max(pred) for pred in mc_all_preds_sample]
            mc_entropies = [-(pred * np.log(pred + 1e-8)).sum() for pred in mc_all_preds_sample]

            sample_data = {
                'sample_id': i,
                'true_label': int(results['true_labels'][i]),
                'is_oos': bool(results['is_oos'][i]),

                # M1 Results
                'm1_prediction': int(results['m1_predictions'][i]),
                'm1_max_confidence': float(np.max(results['m1_probabilities'][i])),
                'm1_entropy': float(results['m1_entropy'][i]),

                # M2 Average Results
                'm2_prediction': int(results['m2_predictions'][i]),
                'm2_max_confidence': float(results['m2_max_conf'][i]),
                'm2_entropy': float(results['mc_entropy'][i]),
                'm2_variance': float(results['mc_variance'][i]),

                # M2 All 50 Results
                'm2_all_confidences': [float(x) for x in mc_confidences],
                'm2_all_entropies': [float(x) for x in mc_entropies],
            }
            detailed_data.append(sample_data)

        return detailed_data

    def print_summary(self, metrics):
        """Print comparison table."""
        print("\n" + "="*70)
        print("COMPLETE RESULTS")
        print("="*70)

        if 'intent_accuracy' in metrics:
            print(f"\nIntent Accuracy: M1={metrics['intent_accuracy']['m1']:.1%}, M2={metrics['intent_accuracy']['m2']:.1%}")

        if 'ood_detection' in metrics:
            print("\nOOD Detection:")
            print(f"{'Method':<5} {'Metric':<15} {'AUROC':<10} {'AUPR':<10}")
            print("-"*70)
            for name, scores in metrics['ood_detection'].items():
                method = name.split('_')[0].upper()
                metric = '_'.join(name.split('_')[1:])
                print(f"{method:<5} {metric:<15} {scores['auroc']:<10.1%} {scores['aupr']:<10.1%}")