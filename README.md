# Triagegeist: Predicting Emergency Triage Acuity with a Multimodal AI Pipeline

LightGBM ensemble with NLP fusion, conformal uncertainty quantification,
and demographic bias auditing, benchmarked against the NEWS2 clinical scoring system.

## Results
- OOF QWK: 0.9914 vs NEWS2 baseline 0.7723 (+0.2191 improvement)
- OOF Accuracy: 0.9827 vs NEWS2 baseline 0.4076

## How to Run
1. Open the notebook in Kaggle at [link to your kaggle notebook]
2. Dataset is the Triagegeist competition dataset at kaggle.com/competitions/triagegeist
3. Run all cells top to bottom
4. Random seed is fixed at 42 throughout

## Requirements
All standard Kaggle environment packages. No GPU required.
lightgbm, shap, scikit-learn, pandas, numpy, matplotlib, seaborn
