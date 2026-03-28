# Triagegeist: Robust Clinical Triage AI Pipeline

## Overview
Triagegeist is an end-to-end machine learning pipeline designed for Emergency Department (ED) triage classification. 

In high-stakes healthcare environments, optimizing a model solely for raw accuracy is insufficient and potentially dangerous. Clinical deployment requires trust, transparency, and an understanding of the model's limitations. To bridge this gap, Triagegeist is engineered with advanced uncertainty quantification, deep model interpretability, and strict algorithmic fairness auditing. It provides healthcare professionals with reliable, equitable, and transparent triage recommendations.

## Core Architecture
The predictive engine utilizes Tree-Based Gradient Boosting (specifically XGBoost and LightGBM). This architecture was selected for its superior performance on tabular clinical data, effectively processing structured inputs such as patient vital signs, demographics, and initial triage nurse assessments without requiring complex deep learning architectures that are harder to interpret.

## Key Features & Pipeline Analysis

### 1. Model Performance and Calibration
While the model achieves robust discrimination across all triage acuity levels, its primary strength lies in its calibration. Triagegeist ensures that the predicted probabilities align strictly with real-world observed frequencies. For instance, if the model predicts a 15% probability of a patient requiring immediate resuscitation, the actual historical frequency for that patient profile is consistently near 15%. This calibration prevents overconfident but incorrect predictions.

### 2. Clinical Interpretability (SHAP)
To eliminate the "black box" problem inherent in many AI systems, the pipeline integrates SHapley Additive exPlanations (SHAP). SHAP dynamically maps exactly how each feature influences the model's final decision. By explicitly showing how vital signs (e.g., NEWS2 scores, heart rate, or oxygen saturation) push the prediction toward a specific triage level, the model allows attending physicians to verify the clinical logic behind every recommendation.

### 3. Uncertainty Quantification (Conformal Prediction)
Standard machine learning models output point predictions, which force the AI to guess even when it is uncertain. Triagegeist implements Conformal Prediction to solve this. Instead of a single triage level, the model outputs a statistically rigorous "prediction set." We guarantee that the true triage level is contained within this set with a user-defined confidence level (e.g., 95%). If a patient's presentation is highly ambiguous, the prediction set naturally widens, signaling to the physician that human judgment must take precedence.

### 4. Algorithmic Bias and Fairness Auditing
Machine learning models trained on historical hospital data run the risk of inheriting and amplifying human biases. To guarantee equitable healthcare delivery, Triagegeist executes an automated fairness audit. It runs strict parity and disparate impact checks, slicing the performance metrics across protected demographic groups (such as race, gender, and age). This ensures the algorithm does not systematically under-triage specific populations.

### 5. Human-in-the-Loop Benchmarking (Nurse Variability)
An AI's performance cannot be evaluated in a vacuum; it must be contextualized against existing clinical workflows. Triagegeist includes an analysis module that quantifies historical triage nurse variability, measuring how often human nurses disagree on the same patient profile. By benchmarking the AI's predictions against this baseline, we mathematically demonstrate that the model's error rate is highly competitive with, or superior to, standard human disagreement in the Emergency Department.

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/YourUsername/Triagegeist.git](https://github.com/YourUsername/Triagegeist.git)
   cd Triagegeist
