# Predicting Diabetes Risk from Self-Reported Health Indicators

**Traditional machine learning vs. deep learning on the CDC BRFSS 2015 survey (UCI dataset 891)**

| Resource | Link |
|---|---|
| Notebook | [`diabetes_risk_prediction.ipynb`](diabetes_risk_prediction.ipynb) · [Open in Colab](https://colab.research.google.com/github/Shumbusho43/SummativeML/blob/main/diabetes_risk_prediction.ipynb) |
| Google Drive (raw data + saved models) | https://drive.google.com/drive/folders/1cvrziKSbJSD7fCHvzdFkXYAlizxkadWu?usp=sharing|
| Video presentation | [Watch on YouTube](https://youtu.be/gv-O7atbpes) |
| Report (IEEE) | [PDF](report/diabetes_risk_prediction_report.pdf) · [Word](report/diabetes_risk_prediction_report.docx) · [Markdown source](report/diabetes_risk_prediction_report.md) |

## Problem

About half of adults with type 2 diabetes in sub-Saharan Africa are undiagnosed, because laboratory screening is expensive and not always available. This project asks whether a **short questionnaire alone**, with no blood test, can identify people who should be referred for screening. It is framed as binary classification: `0 = no diabetes`, `1 = prediabetes or diabetes`.

## Dataset

[CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators) (UCI Machine Learning Repository). It is derived from the 2015 Behavioral Risk Factor Surveillance System telephone survey.

- **253,680 rows × 21 features.** After removing 24,206 exact duplicates, 229,474 rows remain.
- The features are 14 binary indicators (blood pressure, cholesterol, smoking, physical activity and others), 4 ordinal scales (general health, age band, education, income) and 3 counts (BMI, poor mental-health days, poor physical-health days).
- The target is imbalanced: **13.9% positive**.

The notebook downloads the CSV from UCI automatically. **The raw data is not committed to this repository.** A zipped copy is in the Google Drive folder.

## Methodology

| Stage | What was done |
|---|---|
| EDA | Value ranges, duplicates, class balance, class-conditional distributions, prevalence gradients, correlation analysis |
| Cleaning | De-duplication *before* splitting (prevents leakage); BMI clipped to [12, 60] |
| Feature engineering | 9 domain features: cardiometabolic load, lifestyle score, unhealthy days, WHO BMI bands, Age×BMI, socio-economic index, care barrier, poor-health and obesity flags |
| Splits | Stratified 70 / 15 / 15 train / validation / test, shared by all models |
| Traditional ML | Dummy, Gaussian NB, Logistic Regression, Linear SVM (Platt), Random Forest, Histogram Gradient Boosting. Experiments: feature-engineering ablation, class-weighting study, `RandomizedSearchCV` tuning, learning curves |
| Deep learning | Keras models fed by `tf.data` pipelines (`cache → shuffle → batch → prefetch`). Sequential MLP baselines; a deep MLP with BatchNorm, Dropout and L2; a **Functional wide-and-deep model with embeddings**; a random hyperparameter search over architecture, dropout, learning rate, optimiser and batch size |
| Evaluation | ROC and PR curves; confusion matrices at three operating points (0.5, F1-optimal, recall ≥ 80%); calibration curves; bootstrap 95% CIs; paired bootstrap test of ML vs DL |
| Error analysis | FN/FP profiling, subgroup performance by age, sex, income and BMI, permutation importance, inspection of the most confident errors |

## Key results (held-out test set, n = 34,422; Google Colab run)

| Model | ROC-AUC [95% CI] | PR-AUC | Recall / Specificity @ recall ≥ 80% threshold |
|---|---|---|---|
| Logistic Regression (tuned) | 0.810 [0.804, 0.816] | 0.417 | 79.9% / 67.5% |
| Hist Gradient Boosting (tuned) | 0.817 [0.811, 0.823] | 0.438 | 79.5% / 68.4% |
| Keras MLP baseline (DL-2) | 0.817 [0.812, 0.823] | 0.437 | 79.6% / 68.7% |
| Best Keras model (DL-4: Functional wide & deep with embeddings) | 0.817 [0.812, 0.823] | 0.438 | 79.6% / 68.7% |

- **Traditional ML and deep learning are statistically tied.** The paired-bootstrap ΔAUC between tuned gradient boosting and the best Keras model is −0.0005 (95% CI −0.0016 to +0.0005), an interval that includes 0. Gradient boosting gets there with default settings and about 2 seconds of training.
- **The ceiling is set by the data, not the models.** Learning curves plateau, extra network capacity adds nothing, and twelve DL configurations span less than 0.005 AUC.
- **Imbalance handling and threshold choice matter far more than model choice.** They move recall between about 16% and 80% while leaving AUC unchanged.
- **Error analysis:** missed cases are diabetics *without* the usual risk footprint (normal BMI, no hypertension, good self-rated health). Recall is much lower for younger and leaner respondents.

All result tables are in [`results/`](results/) and all figures in [`figures/`](figures/).

## Repository structure

```
├── diabetes_risk_prediction.ipynb   # main notebook (documented, with outputs)
├── diabetes_features.py             # feature definitions (imported by the notebook and saved pipeline)
├── requirements.txt
├── figures/                         # all figures used in the report (PNG, 200 dpi)
├── results/                         # CSV tables for every experiment
└── (generated, git-ignored)
    ├── data/                        # raw CSV, downloaded at run time
    ├── models/                      # best_ml_pipeline.joblib, best_dl_model.keras, preprocessor, model_card.json
    └── drive_upload/                # zipped raw dataset + models, ready for Google Drive
```

## How to run

**Google Colab (recommended).** Open the notebook with the Colab badge above and choose *Runtime → Run all*. scikit-learn and TensorFlow are pre-installed. A full run takes roughly 25–35 minutes on a standard CPU runtime. Set `FAST_MODE = True` in the setup cell for a 3-minute smoke test on a 25k-row sample.

**Locally**

```bash
python -m venv .venv
```
```bash
source .venv/bin/activate
```
```bash
pip install -r requirements.txt
```
```bash
jupyter notebook diabetes_risk_prediction.ipynb
```

Results are reproducible: every random process is seeded with `SEED = 42`. scikit-learn results are bit-for-bit identical across runs. Keras results can vary in the third decimal place between hardware and TensorFlow versions.

## Using the saved models

```python
import joblib, pandas as pd
from tensorflow import keras
import diabetes_features  # must be importable: the pipeline references engineer_features

ml = joblib.load("models/best_ml_pipeline.joblib")           # raw survey columns in, probability out
X = pd.DataFrame([{...21 raw survey columns...}])
risk = ml.predict_proba(X)[:, 1]

# The best DL model in the Colab run is the wide-and-deep network, which takes a dict of named inputs
dl = keras.models.load_model("models/best_dl_model.keras")
prep = joblib.load("models/dl_wide_deep_preprocessor.joblib")
inputs = {"dense_features": prep.transform(X).astype("float32")}
inputs.update({c: X[c].to_numpy("int32").reshape(-1, 1) for c in ["Age", "GenHlth", "Income", "Education"]})
risk_dl = dl.predict(inputs).ravel()
```

Decision thresholds and test metrics for both models are recorded in `models/model_card.json`. Predicted probabilities come from class-weighted models and **over-estimate absolute risk**, so treat them as rankings or use the recorded thresholds.

## Limitations

The data are self-reported and cross-sectional, and come from U.S. adults in 2015. The model therefore detects existing disease and its consequences rather than predicting future onset. It would need recalibration and validation on local data before any use in an African screening setting. See Section 10 of the notebook for the full discussion.
