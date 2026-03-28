# Triagegeist: Robust Clinical Triage AI Pipeline

An end-to-end machine learning pipeline for Emergency Department (ED) triage classification. 

While most clinical ML models optimize solely for raw accuracy, Triagegeist is engineered for clinical trust. This pipeline implements advanced uncertainty quantification (Conformal Prediction), deep model interpretability (SHAP), and algorithmic fairness auditing to ensure safe deployment in high-stakes healthcare environments.

## Key Architecture

* **Tree-Based Gradient Boosting:** Optimized predictive engine for tabular clinical data (vitals, demographics, triage nurse assessments).
* **Conformal Prediction:** Wraps the base model to output statistically rigorous prediction sets rather than highly confident, but potentially incorrect, point predictions.
* **Algorithmic Fairness Audit:** Automated slicing to detect performance disparities across protected demographic groups.
* **Human-in-the-Loop Analysis:** Quantifies baseline triage nurse variability to benchmark the AI's performance against actual human baselines.

## Pipeline Results & Interpretability

### 1. Model Performance & Calibration
The core model achieves robust discrimination across triage acuity levels. Calibration ensures that predicted probabilities align with real-world observed frequencies.

![Final Results](final_results.png)

### 2. Clinical Interpretability (SHAP)
To avoid the black box problem, we use SHapley Additive exPlanations (SHAP) to map exactly how vital signs (e.g., NEWS2 scores, heart rate) and clinical inputs drive the model's triage decisions.

![SHAP Analysis](shap_analysis.png)

### 3. Uncertainty Quantification (Conformal Analysis)
In clinical settings, knowing when a model is unsure is critical. Our conformal prediction layer guarantees that the true triage level is contained within the prediction set with a user-defined confidence level (e.g., 95%).

![Conformal Analysis](conformal_analysis.png)

### 4. Algorithmic Bias & Fairness Audit
Models trained on historical hospital data can inherit human biases. We run strict parity and disparate impact audits across subgroups to ensure equitable triage recommendations.

![Bias Audit](bias_audit.png)

### 5. Benchmarking Nurse Variability
To contextualize the AI's performance, we analyze historical triage nurse variability, proving that the model's error rate is competitive with, or superior to, baseline human disagreement in the ED.

![Nurse Variability](nurse_variability.png)

## Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/Triagegeist.git](https://github.com/YourUsername/Triagegeist.git)
   cd Triagegeist
