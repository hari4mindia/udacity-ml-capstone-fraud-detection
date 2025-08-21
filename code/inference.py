import json, os, numpy as np
from joblib import load

def model_fn(model_dir):
    return load(os.path.join(model_dir, "model.joblib"))

def input_fn(request_body, content_type):
    data = json.loads(request_body)
    # Expecting { "features": [[...30 numbers...], ...] }
    return np.array(data["features"], dtype=float)

def predict_fn(input_data, model):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(input_data)[:, 1]
    return model.predict(input_data)

def output_fn(prediction, accept):
    return json.dumps({"fraud_probability": prediction.tolist()})
