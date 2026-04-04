# Triagegeist: Clinical AI Pipeline for Emergency Triage Acuity Prediction

A stacked ensemble clinical decision support system for predicting Emergency Severity Index (ESI) levels in emergency department triage. Built for the [Triagegeist Competition](https://www.kaggle.com/competitions/triagegeist) hosted by the Laitinen-Fredriksson Foundation.

**Final OOF QWK: 0.9989 | +0.2266 over NEWS2 clinical baseline**

---

## Overview

Emergency department triage nurses assign ESI acuity levels (1–5) under extreme cognitive load with incomplete information. Inter-rater variability (kappa 0.60–0.80) and systematic undertriage of vulnerable populations are well-documented patient safety concerns. This project builds an AI-powered second opinion that is fast, consistent, and auditable.

### Key Features

- **Stacked Ensemble Architecture** — LightGBM + XGBoost + CatBoost base learners with a logistic regression meta-learner
- **Dual-Channel NLP** — Word-level (1–3 grams) and character-level (2–5 grams) TF-IDF on chief complaint text
- **Conformal Prediction** — Distribution-free prediction sets with guaranteed 90% coverage for uncertainty quantification
- **Clinical Cost Analysis** — Asymmetric cost matrix reflecting that undertriage is far more dangerous than overtriage
- **Demographic Bias Audit** — Chi-squared significance testing across sex, insurance, language, and age group
- **Ablation Study** — Systematic feature group removal quantifying each component's contribution
- **SHAP Interpretability** — Global and per-class feature explanations for clinical auditability
- **Nurse-Level Variability Analysis** — ANOVA-based inter-rater variability assessment across 50 triage nurses

---

## Results

| Model | OOF QWK |
|-------|---------|
| LightGBM | 0.9988 |
| XGBoost | 0.9988 |
| CatBoost | 0.9975 |
| Weighted Average | 0.9987 |
| **Stacked Meta-Learner** | **0.9989** |
| NEWS2 Baseline | 0.7723 |

- **416 engineered features** from 4 source files
- **99.78% accuracy** on out-of-fold predictions
- **No statistically significant bias** across any demographic group (all p > 0.05)
- **98.8% empirical coverage** at 90% conformal prediction target
- NLP features identified as the single most critical component (ablation Δ = -0.0642 QWK)

---

## Repository Structure

```
.
├── triagegeist-clinical-ai-pipeline.ipynb   # Main notebook (runs end-to-end on Kaggle)
├── README.md                                 # This file
└── submission.csv                            # Test set predictions (20,000 patients)
```

---

## Pipeline Architecture

```
Input Data (4 tables)
    │
    ▼
Feature Engineering (416 features)
    ├── Vital sign flags & missingness indicators
    ├── Comorbidity burden & high-risk clusters
    ├── Temporal features (cyclical hour, weekend)
    ├── High-risk keyword flags (15 patterns)
    └── Dual-channel TF-IDF (200 word + 100 char)
    │
    ▼
Level-1: Base Models (5-fold stratified CV)
    ├── LightGBM  (leaf-wise, QWK: 0.9988)
    ├── XGBoost   (level-wise, QWK: 0.9988)
    └── CatBoost  (ordered boosting, QWK: 0.9975)
    │
    ▼
Level-2: Stacking Meta-Learner
    └── Logistic Regression on 15-dim OOF probability space
    │
    ▼
Post-Prediction Analysis
    ├── Conformal prediction (90% coverage sets)
    ├── Clinical cost analysis (asymmetric cost matrix)
    ├── SHAP interpretability (global + ESI-1 specific)
    ├── Demographic bias audit (chi-squared tests)
    ├── Nurse-level variability (ANOVA)
    └── Ablation study (feature group contributions)
```

---

## How to Run

### On Kaggle (Recommended)

1. Go to the [notebook on Kaggle](https://www.kaggle.com/code/dhruvjain35/triagegeist-clinical-ai-pipeline)
2. Click **Copy & Edit**
3. Ensure the Triagegeist competition dataset is attached
4. Click **Run All**
5. Runtime: ~2.5 hours on T4 GPU

### Locally

```bash
# Clone the repository
git clone https://github.com/Dhruvjain35/triagegeist-clinical-ai.git
cd triagegeist-clinical-ai

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost shap scipy

# Download the dataset from Kaggle
kaggle competitions download -c triagegeist
unzip triagegeist.zip -d data/

# Update PATH in the notebook to point to your local data directory
# Then run the notebook
jupyter notebook triagegeist-clinical-ai-pipeline.ipynb
```

---

## Dataset

- **Source:** Triagegeist Synthetic ED Dataset, Laitinen-Fredriksson Foundation
- **Access:** [kaggle.com/competitions/triagegeist/data](https://kaggle.com/competitions/triagegeist/data)
- **License:** Non-Commercial Research License
- **Files:** `train.csv` (80,000 patients), `test.csv` (20,000 patients), `chief_complaints.csv`, `patient_history.csv`
- **Note:** All records are fully synthetic. No real patient data is used.

---

## Key Findings

1. **NLP is critical** — Removing all NLP features drops QWK by 0.0642. Character-level TF-IDF alone contributes 0.0048, validating the dual-channel approach for capturing misspellings and abbreviations in triage nurse documentation.

2. **Pain score is the strongest single predictor** — Consistent with ESI's resource-based classification where pain drives resource utilization estimates.

3. **NEWS2 and GCS dominate ESI-1 predictions** — SHAP analysis shows GCS total is the primary driver for critical (ESI-1) predictions, aligning with the ESI algorithm's emphasis on altered consciousness.

4. **No demographic bias detected** — Chi-squared tests show no significant undertriage differences across sex, insurance, language, or age group. The Somali language subgroup shows a marginally elevated critical miss rate (2.0%) warranting prospective monitoring.

5. **Stacking outperforms averaging** — The meta-learner improves QWK by +0.0002 over weighted averaging, with the largest gains on borderline ESI 1/2 cases.

---

## Limitations

- Synthetic dataset — requires validation on real clinical data (e.g., MIMIC-IV-ED)
- NEWS2 as a pre-computed feature partially encodes existing triage logic
- TF-IDF misses semantic nuance — clinical language models would improve NLP
- No temporal validation (time-based splits would better simulate deployment)
- Single-snapshot prediction without reassessment dynamics
- No external validation on held-out institutions

---

## Competition

- **Competition:** [Triagegeist — Laitinen-Fredriksson Foundation](https://www.kaggle.com/competitions/triagegeist)
- **Task:** Predict ESI triage acuity (1–5) from ED intake data
- **Evaluation:** Judged by clinical relevance, technical quality, documentation, insights, and novelty
- **Prize Pool:** $10,000

---

## Citation

```
Olaf Yunus Laitinen Imanov (2026). Triagegeist. Kaggle.
https://kaggle.com/competitions/triagegeist
```

---

## License

This project is for non-commercial and academic research use only, per the competition data license.
