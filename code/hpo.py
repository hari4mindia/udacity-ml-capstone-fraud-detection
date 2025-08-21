# You can run this file as a script in SageMaker Studio's terminal or paste it into a notebook cell.
import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.image_uris import retrieve
from sagemaker.tuner import HyperparameterTuner, ContinuousParameter, IntegerParameter

# --------- EDIT THESE ---------
ROLE   = "<YOUR_SAGEMAKER_EXECUTION_ROLE_ARN>"
BUCKET = "<YOUR_S3_BUCKET_NAME>"
PREFIX = "fraud"
# ------------------------------

session = sagemaker.Session()
region = session.boto_region_name

container = retrieve("xgboost", region, version="1.5-1")
xgb = Estimator(
    image_uri=container,
    role=ROLE,
    instance_count=1,
    instance_type="ml.m5.large",
    output_path=f"s3://{BUCKET}/{PREFIX}/outputs/",
    hyperparameters={
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "verbosity": 1,
        # Adjust to your class ratio (non-fraud / fraud) ~ 284315/492 ≈ 577
        "scale_pos_weight": 577
    },
)

hp_ranges = {
    "max_depth": IntegerParameter(3, 10),
    "eta": ContinuousParameter(0.01, 0.3),
    "subsample": ContinuousParameter(0.5, 1.0),
    "colsample_bytree": ContinuousParameter(0.5, 1.0),
    "min_child_weight": IntegerParameter(1, 10),
}

tuner = HyperparameterTuner(
    estimator=xgb,
    objective_metric_name="validation:aucpr",
    hyperparameter_ranges=hp_ranges,
    max_jobs=20,
    max_parallel_jobs=4,
    objective_type="Maximize",
)

train_in = TrainingInput(f"s3://{BUCKET}/{PREFIX}/data/processed/train.csv", content_type="text/csv")
val_in   = TrainingInput(f"s3://{BUCKET}/{PREFIX}/data/processed/val.csv",   content_type="text/csv")

tuner.fit({"train": train_in, "validation": val_in})
print("Launched HPO job. Track progress in SageMaker > Hyperparameter tuning jobs.")
