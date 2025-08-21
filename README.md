# Credit Card Fraud Detection (SageMaker Capstone)

## How to Run (Quick)
1. Open `notebooks/data_exploration.ipynb` → set REGION/BUCKET → preprocess and upload `train/val/test` to S3.
2. Open `notebooks/train_and_deploy.ipynb` → set ROLE/REGION/BUCKET → run baseline training. Note `est.model_data`.
3. (Optional) Launch HPO: run `code/hpo.py` (edit ROLE/BUCKET) or paste into a notebook cell.
4. Open `notebooks/evaluation.ipynb` → set `MODEL_DATA_S3` to your artifact → compute metrics + plots.
5. Open `notebooks/deploy_model.ipynb` → set `MODEL_DATA_S3` → deploy endpoint and test JSON inference.
6. (Optional) Enable data capture & Model Monitor from console. Export Profiler report from training job.

## Files
- `code/train.py` – baseline Logistic Regression training (class_weight balanced).
- `code/hpo.py` – XGBoost hyperparameter tuning using SageMaker built-in container.
- `code/inference.py` – real-time inference script for endpoint.
- `code/utils.py` – helper to load CSVs from channels.
- `notebooks/*.ipynb` – EDA, training, evaluation, deployment.
- `fraud_capstone_report_template.pdf` – report template aligned to rubric.

## Primary Metrics
- PR-AUC (primary), ROC-AUC, Recall, Precision, F1, Confusion Matrix.

## Notes
- Dataset: Kaggle Credit Card Fraud Detection (284,807 rows; 492 fraud ~0.172%).
- Scale `Amount` and `Time`; PCA features `V1..V28` already standardized.
- For XGBoost, set `scale_pos_weight` ≈ non-fraud/fraud (~577).

## Requirements
- SageMaker Studio (or Notebook), IAM role with S3 and SageMaker permissions.
- Python 3.9+, `sagemaker`, `boto3`, `scikit-learn`, `pandas`, `matplotlib`.
