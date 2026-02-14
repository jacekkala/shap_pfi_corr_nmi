import shap
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from SHAP_utils.load_breast_cancer import load_breast_cancer_dataset
from shap_store import save_shap

data = load_breast_cancer_dataset(r"PATH_TO_DATA_FOLDER")
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test_s = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_s, y_train)

explainer_lr = shap.LinearExplainer(
    lr,
    X_train_s,
    feature_perturbation="interventional"
)
shap_lr = explainer_lr(X_test_s)

lr_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': lr.coef_.flatten(),
    'Abs_Coefficient': np.abs(lr.coef_.flatten())
}).sort_values(by='Abs_Coefficient', ascending=False)

save_shap(
    path="shap_results/breast_cancer_logreg.pkl",
    shap_values=shap_lr,
    X_test=X_test_s,
    model_name="logistic_regression",
    dataset_name="breast_cancer",
    explainer_name="LinearExplainer",
    extra_meta={"coefficients": lr_importance}
)

xgb_model = xgb.XGBClassifier(
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

explainer_xgb = shap.TreeExplainer(
    xgb_model,
    X_train,
    feature_perturbation="interventional"
)
shap_xgb = explainer_xgb(X_test)

xgb_gain_dict = xgb_model.get_booster().get_score(importance_type='gain')
xgb_importance = pd.DataFrame(
    list(xgb_gain_dict.items()), columns=['Feature', 'Gain']
).sort_values(by='Gain', ascending=False)

save_shap(
    path="shap_results/breast_cancer_xgboost.pkl",
    shap_values=shap_xgb,
    X_test=X_test,
    model_name="xgboost",
    dataset_name="breast_cancer",
    explainer_name="TreeExplainer",
    extra_meta={"gain": xgb_importance}
)
