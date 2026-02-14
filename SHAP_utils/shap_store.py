import pickle
from pathlib import Path

def save_shap(
    path,
    shap_values,
    X_test,
    model_name,
    dataset_name,
    explainer_name,
    extra_meta=None
):
    payload = {
        "shap_values": shap_values.values,
        "base_values": shap_values.base_values,
        "data": X_test.values,
        "feature_names": list(X_test.columns),
        "n_samples": X_test.shape[0],
        "model": model_name,
        "dataset": dataset_name,
        "explainer": explainer_name,
        "extra_meta": extra_meta or {}
    }

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_shap(path):
    with open(path, "rb") as f:
        return pickle.load(f)
