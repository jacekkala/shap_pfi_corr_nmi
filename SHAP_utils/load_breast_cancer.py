import pandas as pd
import os

def load_breast_cancer_dataset(data_folder):
    base_features = [
        "radius", "texture", "perimeter", "area", "smoothness",
        "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"
    ]

    columns = ["id", "diagnosis"]
    for suffix in ["_mean", "_se", "_worst"]:
        for feature in base_features:
            columns.append(f"{feature}{suffix}")

    data_path = os.path.join(data_folder, "wdbc.data")
    
    try:
        df = pd.read_csv(data_path, header=None, names=columns)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return None

    return df

# df = load_breast_cancer_dataset(r"PATH_TO_DATA_FOLDER")
# df.head()