# Out-of-Domain Detection: Monte Carlo Dropout vs Standard Inference

## Research Question

Does Monte Carlo Dropout improve out-of-domain detection over standard inference with fine-tuned DistilBERT when using comparable uncertainty metrics on CLINC150?

## Google Colab Notebook

🔗 **[Run in Colab](<https://colab.research.google.com/drive/1SWbUartK0CBQrUobg9CNAazeupsMBI6o?usp=sharing>)**

## Dataset

CLINC150: Intent classification with out-of-domain detection
- In-domain: 150 intent classes
- Training set: 15250 samples
- Validation set: 3100 samples
- Test set: 5500 samples (18.2% OOS)

## Methods

**M1 (Standard Inference):** Single forward pass, dropout OFF  
**M2 (Monte Carlo Dropout):** 50 forward passes, dropout ON

**Comparable Metrics:**
- Inverse Confidence: 1 - max(softmax)
- Entropy: uncertainty in the mean distribution

**M2-Only Metric:**
- Variance: spread across 50 predictions

## Results

### Key Finding

**Monte Carlo Dropout provides no benefit for OOD detection on CLINC150.** Comparing identical metrics across methods shows M1 and M2 differ by only 0.2%, making the 50× computational cost unjustified.

| Metric | M1 (AUPR) | M2 (AUPR) | Difference |
|--------|-----------|-----------|------------|
| Inverse Confidence | 87.8% | 87.6% | -0.2% |
| Entropy | 89.3% | 89.1% | -0.2% |
| Variance | N/A | 40.1% | M2-only |

### Additional Findings

1. **Metric selection matters more than method:** Entropy outperforms inverse confidence by 1.5% within each method 

2. **Variance fails:** Achieves only 40.1% AUPR, demonstrating poor discriminative power for intent classification OOD detection.

3. **Intent accuracy preserved:** M1 (95.2%) vs M2 (95.3%) — negligible difference.

## Conclusion

For well-trained transformer models on intent classification, expensive uncertainty estimation methods provide no advantage over simple confidence baselines. **Recommendation:** Use entropy-based uncertainty from standard inference.

## File Structure
```
ood_detection_clinc150/
├── config.py              # Configuration parameters
├── data_loader.py         # CLINC150 dataset loading
├── model.py               # DistilBERT classifier
├── trainer.py             # Training logic
├── evaluator.py           # M1 vs M2 evaluation
├── utils.py               # Helper functions
├── visualization.py       # Result visualization
└── main.py                # Pipeline orchestration
```

## References

GeeksforGeeks. What is Monte Carlo (MC) Dropout? https://www.geeksforgeeks.org/deep-learning/what-is-monte-carlo-mc-dropout/, 2024. Accessed: 2025-12-07.

Jurafsky, Daniel and James H. Martin. Speech and Language Processing (3rd ed. draft). https://web.stanford.edu/~jurafsky/slp3/ed3book.pdf, 2023. Accessed: 2025-12-07.

Larson, Stefan, et al. CLINC150: An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction. Hugging Face. https://huggingface.co/datasets/clinc/clinc_oos. Accessed: 2025-12-07.

Jerry A. " BERT Transformer Explained Visually | [CLS] Token, Embeddings & Pooled Output Simplified. " YouTube video, 10:43, 2025. https://www.youtube.com/watch?v=s08A72X__Hg. Accessed: 2025-12-07.

Musawi, Rahmat. "DistilBERT Finetuned Emotion." Medium. https://medium.com/@musawi.rahmat/distilbert-finetuned-emotion-368237455a96. Accessed: 2025-12-07.

Claude (Anthropic). AI assistant. https://claude.ai. Accessed: 2025-12-07.