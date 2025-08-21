import argparse, os, json
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_fscore_support
from utils import load_csv_dir

def main(args):
    Xtr, ytr = load_csv_dir(args.train, "train.csv")
    Xv,  yv  = load_csv_dir(args.val,   "val.csv")

    # Baseline model with class weights for imbalance
    clf = LogisticRegression(max_iter=600, class_weight="balanced", n_jobs=-1)
    clf.fit(Xtr, ytr)

    # Validation metrics
    proba = clf.predict_proba(Xv)[:, 1]
    preds = (proba >= 0.5).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(yv, preds, average="binary", zero_division=0)
    metrics = {
        "pr_auc": float(average_precision_score(yv, proba)),
        "roc_auc": float(roc_auc_score(yv, proba)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "threshold": 0.5
    }
    print(json.dumps({"validation_metrics": metrics}))

    os.makedirs(args.model_dir, exist_ok=True)
    dump(clf, os.path.join(args.model_dir, "model.joblib"))
    with open("/opt/ml/output/metrics.json", "w") as f:
        json.dump(metrics, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--val",   default=os.environ.get("SM_CHANNEL_VAL"))
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()
    main(args)
