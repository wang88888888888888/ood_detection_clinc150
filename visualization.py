# file 7: visualization.py
"""
Visualization module for NLP-OOD project
Comprehensive visualization functions, including data saving, uncertainty comparison plots, MC dropout variation plots, etc.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set font size and style
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.figsize'] = (12, 8)

class ResultsVisualizer:
    """
    Results visualization class - Fang's implementation
    Features:
    1. Save detailed sample-by-sample results to JSON/CSV
    2. Plot 5 uncertainty score comparison lines
    3. Show 50 MC dropout variations
    4. AUROC/AUPR explanation plot
    """

    def __init__(self, results_data):
        """
        Initializes the visualizer
        Args:
            results_data: Detailed results obtained from the evaluator
        """
        self.data = results_data
        self.colors = {
            'm1_max_conf': '#1f77b4',    # Blue
            'm1_entropy': '#ff7f0e',     # Orange
            'm2_max_conf': '#2ca02c',    # Green
            'm2_entropy': '#d62728',     # Red
            'm2_variance': '#9467bd'     # Purple
        }

    def save_detailed_results(self, filename='v5_detailed_results.json', save_csv=True):
        """
        Saves detailed sample-by-sample results
        Args:
            filename: JSON filename
            save_csv: Whether to also save a CSV version
        """
        print(f"💾 Saving detailed results to {filename}...")

        # Save in JSON format
        with open(filename, 'w') as f:
            json.dump(self.data, f, indent=2, default=str)

        if save_csv:
            # Convert to DataFrame and save CSV
            df_data = []
            for sample in self.data:
                row = {
                    'sample_id': sample['sample_id'],
                    'true_label': sample['true_label'],
                    'is_oos': sample['is_oos'],
                    'm1_prediction': sample['m1_prediction'],
                    'm1_max_confidence': sample['m1_max_confidence'],
                    'm1_entropy': sample['m1_entropy'],
                    'm2_prediction': sample['m2_prediction'],
                    'm2_max_confidence': sample['m2_max_confidence'],
                    'm2_entropy': sample['m2_entropy'],
                    'm2_variance': sample['m2_variance'],
                    # MC dropout statistics
                    'm2_conf_std': np.std(sample['m2_all_confidences']),
                    'm2_entropy_std': np.std(sample['m2_all_entropies']),
                }
                df_data.append(row)

            df = pd.DataFrame(df_data)
            csv_filename = filename.replace('.json', '.csv')
            df.to_csv(csv_filename, index=False)
            print(f"   Also saved CSV version: {csv_filename}")

        print(f"   Saved {len(self.data)} samples successfully!")
        return filename

    def plot_uncertainty_comparison(self, max_samples=1000, save_plot=True):
        """
        Plots 5 uncertainty score comparison lines
        Args:
            max_samples: Maximum number of samples to display (to avoid too dense plots)
            save_plot: Whether to save the plot
        """
        print(f" Creating uncertainty comparison plot...")

        # Prepare data
        n_samples = min(len(self.data), max_samples)
        sample_indices = np.linspace(0, len(self.data)-1, n_samples, dtype=int)

        # Extract data
        m1_max_conf = [self.data[i]['m1_max_confidence'] for i in sample_indices]
        m1_entropy = [self.data[i]['m1_entropy'] for i in sample_indices]
        m2_max_conf = [self.data[i]['m2_max_confidence'] for i in sample_indices]
        m2_entropy = [self.data[i]['m2_entropy'] for i in sample_indices]
        m2_variance = [self.data[i]['m2_variance'] for i in sample_indices]

        # Create plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

        # Top plot: Confidence comparison
        ax1.plot(sample_indices, m1_max_conf,
                color=self.colors['m1_max_conf'], linewidth=2,
                label='M1 Max Confidence', alpha=0.8)
        ax1.plot(sample_indices, m2_max_conf,
                color=self.colors['m2_max_conf'], linewidth=2,
                label='M2 Max Confidence', alpha=0.8, linestyle='--')

        ax1.set_ylabel('Confidence Score')
        ax1.set_title('Confidence Comparison: M1 vs M2 (Single vs MC Dropout)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Bottom plot: Entropy + Variance comparison
        ax2.plot(sample_indices, m1_entropy,
                color=self.colors['m1_entropy'], linewidth=2,
                label='M1 Entropy', alpha=0.8)
        ax2.plot(sample_indices, m2_entropy,
                color=self.colors['m2_entropy'], linewidth=2,
                label='M2 Entropy', alpha=0.8, linestyle='--')

        # Use right y-axis for variance (because scale might be different)
        ax2_right = ax2.twinx()
        ax2_right.plot(sample_indices, m2_variance,
                      color=self.colors['m2_variance'], linewidth=2,
                      label='M2 Variance', alpha=0.8, linestyle=':')
        ax2_right.set_ylabel('Variance Score', color=self.colors['m2_variance'])
        ax2_right.tick_params(axis='y', labelcolor=self.colors['m2_variance'])

        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Entropy Score')
        ax2.set_title('Uncertainty Comparison: Entropy vs Variance')
        ax2.legend(loc='upper left')
        ax2_right.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plt.savefig('uncertainty_comparison.png', dpi=300, bbox_inches='tight')
            print("   Saved as uncertainty_comparison.png")

        plt.show()

    def plot_mc_dropout_variations(self, sample_indices=[0, 100, 500], save_plot=True):
        """
        Shows the 50 MC dropout variations for specific samples
        Args:
            sample_indices: Indices of samples to display
            save_plot: Whether to save the plot
        """
        print(f"Creating MC dropout variations plot for samples {sample_indices}...")

        # Ensure valid indices
        valid_indices = [i for i in sample_indices if i < len(self.data)]
        if not valid_indices:
            print("   No valid sample indices provided!")
            return

        n_samples = len(valid_indices)
        fig, axes = plt.subplots(n_samples, 2, figsize=(15, 4*n_samples))
        if n_samples == 1:
            axes = axes.reshape(1, -1)

        for idx, sample_idx in enumerate(valid_indices):
            sample = self.data[sample_idx]

            # Left plot: 50 confidence variations
            ax_conf = axes[idx, 0]
            confidences = sample['m2_all_confidences']
            runs = list(range(1, len(confidences) + 1))

            ax_conf.plot(runs, confidences, 'o-', alpha=0.7, markersize=4)
            ax_conf.axhline(y=sample['m1_max_confidence'],
                           color='red', linestyle='--', linewidth=2,
                           label=f"M1 Confidence: {sample['m1_max_confidence']:.4f}")
            ax_conf.axhline(y=sample['m2_max_confidence'],
                           color='green', linestyle='-', linewidth=2,
                           label=f"M2 Mean: {sample['m2_max_confidence']:.4f}")

            ax_conf.set_xlabel('MC Dropout Run')
            ax_conf.set_ylabel('Max Confidence')
            ax_conf.set_title(f'Sample {sample_idx}: 50 MC Runs - Confidence\n'
                            f'Label: {sample["true_label"]}, OOS: {sample["is_oos"]}')
            ax_conf.legend()
            ax_conf.grid(True, alpha=0.3)

            # Right plot: 50 entropy variations
            ax_entropy = axes[idx, 1]
            entropies = sample['m2_all_entropies']

            ax_entropy.plot(runs, entropies, 's-', alpha=0.7, markersize=4, color='orange')
            ax_entropy.axhline(y=sample['m1_entropy'],
                              color='blue', linestyle='--', linewidth=2,
                              label=f"M1 Entropy: {sample['m1_entropy']:.4f}")
            ax_entropy.axhline(y=sample['m2_entropy'],
                              color='red', linestyle='-', linewidth=2,
                              label=f"M2 Mean: {sample['m2_entropy']:.4f}")

            ax_entropy.set_xlabel('MC Dropout Run')
            ax_entropy.set_ylabel('Entropy')
            ax_entropy.set_title(f'Sample {sample_idx}: 50 MC Runs - Entropy')
            ax_entropy.legend()
            ax_entropy.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_plot:
            plt.savefig('mc_dropout_variations.png', dpi=300, bbox_inches='tight')
            print("   Saved as mc_dropout_variations.png")

        plt.show()

    def plot_score_distributions(self, save_plot=True):
        """
        Plots the score distribution comparison for OOD vs ID samples
        """
        print(f"📈 Creating score distributions plot...")

        # Separate OOD and ID samples
        ood_samples = [s for s in self.data if s['is_oos']]
        id_samples = [s for s in self.data if not s['is_oos']]

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        methods = [
            ('m1_max_confidence', 'M1 Max Confidence'),
            ('m1_entropy', 'M1 Entropy'),
            ('m2_max_confidence', 'M2 Max Confidence'),
            ('m2_entropy', 'M2 Entropy'),
            ('m2_variance', 'M2 Variance')
        ]

        for i, (score_key, title) in enumerate(methods):
            row, col = i // 3, i % 3
            ax = axes[row, col]

            # Get scores
            ood_scores = [s[score_key] for s in ood_samples]
            id_scores = [s[score_key] for s in id_samples]

            # Plot distribution histogram
            ax.hist(id_scores, bins=50, alpha=0.7, label=f'ID (n={len(id_scores)})',
                   color='blue', density=True)
            ax.hist(ood_scores, bins=50, alpha=0.7, label=f'OOD (n={len(ood_scores)})',
                   color='red', density=True)

            ax.set_xlabel(title)
            ax.set_ylabel('Density')
            ax.set_title(f'{title} Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove redundant subplots
        if len(methods) < 6:
            axes[1, 2].remove()

        plt.tight_layout()

        if save_plot:
            plt.savefig('score_distributions.png', dpi=300, bbox_inches='tight')
            print("    Saved as score_distributions.png")

        plt.show()

    def plot_auroc_aupr_explanation(self, method='m1_max_confidence', save_plot=True):
        """
        Explanation plot for AUROC and AUPR
        Args:
            method: The method score to use for demonstration
        """
        print(f" Creating AUROC/AUPR explanation plot using {method}...")

        # Prepare data
        y_true = [int(s['is_oos']) for s in self.data]  # 1 for OOD, 0 for ID

        # Get uncertainty scores (OOD should have higher uncertainty)
        if method.endswith('confidence'):
            # Lower confidence, more likely to be OOD, so take negative value or 1-confidence
            y_scores = [1 - s[method] for s in self.data]
        else:
            # Higher entropy and variance, more likely to be OOD
            y_scores = [s[method] for s in self.data]

        # Calculate ROC and PR curves
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_scores)
        precision, recall, pr_thresholds = precision_recall_curve(y_true, y_scores)

        auroc = auc(fpr, tpr)
        aupr = auc(recall, precision)

        # Create explanation plot
        fig = plt.figure(figsize=(20, 12))

        # 1. Data flow diagram
        ax1 = plt.subplot(2, 4, 1)
        ax1.text(0.1, 0.8, "Step 1: Forward Pass", fontsize=14, weight='bold')
        ax1.text(0.1, 0.7, "Input: 'book a flight'", fontsize=12)
        ax1.text(0.1, 0.6, "Logits: [2.1, -0.5, 3.2, ...]", fontsize=12)
        ax1.text(0.1, 0.5, "Softmax: [0.15, 0.03, 0.82, ...]", fontsize=12)
        ax1.text(0.1, 0.3, "Step 2: Uncertainty Score", fontsize=14, weight='bold')
        ax1.text(0.1, 0.2, f"Max Confidence: 0.82", fontsize=12)
        ax1.text(0.1, 0.1, f"Uncertainty: 1-0.82=0.18", fontsize=12)
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        ax1.set_title('From Predictions to Scores')

        # 2. Threshold illustration
        ax2 = plt.subplot(2, 4, 2)
        thresholds_demo = np.linspace(0, 1, 100)
        ax2.plot(thresholds_demo, thresholds_demo, 'b-', label='ID samples')
        ax2.plot(thresholds_demo, 1-thresholds_demo, 'r-', label='OOD samples')
        ax2.axvline(x=0.5, color='green', linestyle='--', label='Threshold')
        ax2.set_xlabel('Uncertainty Score')
        ax2.set_ylabel('Sample Density')
        ax2.set_title('Threshold Selection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. ROC Curve
        ax3 = plt.subplot(2, 4, 3)
        ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auroc:.3f})')
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curve')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. PR Curve
        ax4 = plt.subplot(2, 4, 4)
        ax4.plot(recall, precision, 'r-', linewidth=2, label=f'PR (AUC = {aupr:.3f})')
        baseline = sum(y_true) / len(y_true)
        ax4.axhline(y=baseline, color='k', linestyle='--', alpha=0.5,
                   label=f'Baseline ({baseline:.3f})')
        ax4.set_xlabel('Recall (True Positive Rate)')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curve')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        # 5-8. Detailed explanations
        explanations = [
            ("AUROC Meaning",
             "• Area Under ROC Curve\n• Measures overall discrimination\n• Higher = better separation\n• Range: 0.5 (random) to 1.0 (perfect)"),
            ("AUPR Meaning",
             "• Area Under PR Curve\n• Focuses on positive class (OOD)\n• Important when classes imbalanced\n• Higher = fewer false alarms"),
            ("When to use AUROC?",
             "• Balanced datasets\n• Care about both FPR and TPR\n• General model comparison\n• Standard metric in ML"),
            ("When to use AUPR?",
             "• Imbalanced datasets\n• Care more about precision\n• Cost of false positives high\n• Rare event detection")
        ]

        for i, (title, text) in enumerate(explanations):
            ax = plt.subplot(2, 4, 5+i)
            ax.text(0.05, 0.95, title, fontsize=14, weight='bold',
                   transform=ax.transAxes, va='top')
            ax.text(0.05, 0.80, text, fontsize=12,
                   transform=ax.transAxes, va='top')
            ax.axis('off')

        plt.tight_layout()

        if save_plot:
            plt.savefig('auroc_aupr_explanation.png', dpi=300, bbox_inches='tight')
            print("    Saved as auroc_aupr_explanation.png")

        plt.show()

        # Print numerical results
        print(f"\n📊 AUROC/AUPR Results for {method}:")
        print(f"   AUROC: {auroc:.4f}")
        print(f"   AUPR:  {aupr:.4f}")
        print(f"   Baseline (random): {baseline:.4f}")

    def generate_all_plots(self):
        """
        Generates all visualization plots
        """
        print(" Generating all visualization plots...")
        print("="*50)

        # 1. Save detailed results
        self.save_detailed_results()

        # 2. Uncertainty comparison plot
        self.plot_uncertainty_comparison()

        # 3. MC dropout variation plot
        self.plot_mc_dropout_variations()

        # 4. Distribution comparison plot
        self.plot_score_distributions()

        # 5. AUROC/AUPR explanation plot
        self.plot_auroc_aupr_explanation()

        print("\n All visualizations completed!")
        print("Generated files:")
        print("   - v5_detailed_results.json")
        print("   - v5_detailed_results.csv")
        print("   - uncertainty_comparison.png")
        print("   - mc_dropout_variations.png")
        print("   - score_distributions.png")
        print("   - auroc_aupr_explanation.png")