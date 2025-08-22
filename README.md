# Fraud Detection Capstone Project

## 📌 Project Overview

This project builds a machine learning-based fraud detection system using the Kaggle Credit Card Fraud dataset. The objective is to accurately classify transactions as fraudulent or legitimate, despite the extreme class imbalance. The solution is deployed using AWS SageMaker and includes data preprocessing, model training, hyperparameter tuning, evaluation, and batch inference.

---

## 📊 Dataset Description

- **Source**: Kaggle Credit Card Fraud Detection
- **Records**: 284,807 transactions
- **Fraud Cases**: 492 (0.172%)
- **Features**: 30 PCA-transformed numerical features, including `Time`, `Amount`, and `Class` (target)

---

## 📁 Project Structure

├── fetch_dataset_ipynb.txt # Downloads and uploads raw dataset to S3 ├── data_exploration_ipynb.txt # Preprocessing, scaling, and train/val/test split ├── train_and_deploy_ipynb 1.txt # Model training, tuning, evaluation, and deployment ├── README.md # Project documentation ├── proposal.pdf # Project proposal ├── Final_CapStone_project_info.docx # Udacity rubric and instructions

---

## ⚙️ Setup Instructions

1. **Environment**: AWS SageMaker Notebook Instance
2. **Dependencies**:
   - `boto3`
   - `sagemaker`
   - `pandas`, `numpy`, `matplotlib`
   - `scikit-learn`
   - `kagglehub`

3. **Kaggle API Setup**:
   - Store credentials in `config/apiToken.env`
   - Format:
     ```
     API_USERNAME=your_username
     API_KEY=your_key
     ```

---

## 🧠 Model Training

### Baseline:
- **Algorithm**: Logistic Regression
- **Framework**: SageMaker SKLearn

### Advanced:
- **Algorithm**: XGBoost
- **Hyperparameters**:
  - `objective`: binary:logistic
  - `eval_metric`: aucpr
  - `scale_pos_weight`: 577
  - `num_round`: 400
- **Tuning**:
  - `max_depth`, `eta`, `subsample`, `colsample_bytree`, `min_child_weight`
  - 12 jobs, 3 parallel

---

## 📈 Evaluation Metrics

| Metric              | Value     |
|---------------------|-----------|
| PR-AUC              | 0.9029    |
| ROC-AUC             | 0.9820    |
| Precision @ 0.5     | 0.8775    |
| Recall @ 0.5        | 0.8775    |
| F1-Score @ 0.5      | 0.8775    |
| Best F1 Threshold   | ~0.386    |
| Best Precision ≥ 0.90 | 0.9111 @ Recall 0.8367 |

---

## 🚀 Deployment Details

- **Model Artifact**:  
  `s3://udacity-fraud-capstone/fraud/outputs/sagemaker-xgboost-250821-1000-002-593cfed0/output/model.tar.gz`

- **Batch Transform Output**:  
  `s3://udacity-fraud-capstone/fraud/batch-preds/test_nolabel.csv.out`

- **Inference**: Batch transform used for scoring test data. Predictions evaluated against true labels.

---

## 🔁 How to Reproduce

1. Run `fetch_dataset_ipynb.txt` to download and upload the dataset to S3.
2. Run `data_exploration_ipynb.txt` to preprocess and split the data.
3. Run `train_and_deploy_ipynb 1.txt` to:
   - Train baseline and XGBoost models
   - Perform hyperparameter tuning
   - Deploy the best model
   - Run batch inference and evaluate predictions

---

## 📬 Author

This project was completed as part of the Udacity Machine Learning Engineer Nanodegree Capstone.