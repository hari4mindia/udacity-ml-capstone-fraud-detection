import os
import pandas as pd

def load_csv_dir(channel_path: str, filename: str):
    """Load CSV from a SageMaker channel directory and split X/y."""
    path = os.path.join(channel_path, filename)
    df = pd.read_csv(path)
    y = df["Class"].values
    X = df.drop(columns=["Class"]).values
    return X, y
