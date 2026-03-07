"""
Evaluating Feature Attribution Stability Using SHAP Across Multiple ML Models.
Loads data, trains 3-4 models, computes SHAP, compares feature importance stability.
"""
import os
import certifi
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())

import argparse
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import shap

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# --- Data loading ---

def load_california(data_path=None):
    if data_path:
        df = pd.read_csv(data_path)
        if "MedHouseVal" not in df.columns:
            raise ValueError("California CSV must include column MedHouseVal")
        y = df["MedHouseVal"]
        X = df.drop(columns="MedHouseVal")
        return X, y, list(X.columns), "regression"
    data = fetch_california_housing(as_frame=True)
    X = data.frame.drop(columns="MedHouseVal")
    y = data.frame["MedHouseVal"]
    return X, y, list(X.columns), "regression"

def load_adult():
    from sklearn.datasets import fetch_openml
    data = fetch_openml("adult", version=2, as_frame=True)
    # OpenML adult uses "class" as target column name
    y = (data.frame["class"] == ">50K").astype(int)
    X = data.frame.drop(columns=["class"])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X = X[num_cols].fillna(X[num_cols].median())
    return X, y, num_cols, "classification"

def load_titanic():
    from sklearn.datasets import fetch_openml
    data = fetch_openml("titanic", version=1, as_frame=True)
    X = data.frame.drop(columns=["survived", "name", "ticket", "cabin", "boat", "body", "home.dest"])
    y = data.frame["survived"].astype(int)
    # Numeric only
    for c in X.columns:
        if X[c].dtype == "object" or X[c].dtype.name == "category":
            X[c] = LabelEncoder().fit_transform(X[c].astype(str))
    X = X.fillna(X.median())
    return X, y, list(X.columns), "classification"

def load_data(name, data_path=None):
    if name == "california":
        return load_california(data_path=data_path)
    if name == "adult":
        return load_adult()
    if name == "titanic":
        return load_titanic()
    raise ValueError(f"Unknown dataset: {name}")


# --- Models ---

def get_models(task, random_state=42):
    if task == "regression":
        models = {
            "RandomForest": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        }
        if HAS_XGB:
            models["XGBoost"] = xgb.XGBRegressor(n_estimators=100, random_state=random_state)
    else:
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        }
        if HAS_XGB:
            models["XGBoost"] = xgb.XGBClassifier(n_estimators=100, random_state=random_state)
    return models


# --- SHAP and stability ---

def mean_abs_shap_importance(explainer, X_background, X_explain):
    """Mean |SHAP| per feature (global importance)."""
    shap_vals = explainer.shap_values(X_explain, check_additivity=False)
    if isinstance(shap_vals, list):
        shap_vals = np.sum(np.abs(shap_vals), axis=0)
    return np.abs(shap_vals).mean(axis=0)

def spearman_rank_correlation(ranks_dict):
    """Pairwise Spearman correlation of feature ranks across models."""
    from scipy.stats import spearmanr
    names = list(ranks_dict.keys())
    n = len(names)
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            r, p = spearmanr(ranks_dict[names[i]], ranks_dict[names[j]])
            results.append((names[i], names[j], r, p))
    return results

def top_k_overlap(ranks_dict, k=5):
    """For each pair of models, Jaccard overlap of top-k feature indices."""
    names = list(ranks_dict.keys())
    n = len(names)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            top_i = set(np.argsort(ranks_dict[names[i]])[:k])
            top_j = set(np.argsort(ranks_dict[names[j]])[:k])
            inter = len(top_i & top_j)
            union = len(top_i | top_j)
            jaccard = inter / union if union else 0.0
            out.append((names[i], names[j], jaccard))
    return out


def main():
    parser = argparse.ArgumentParser(description="SHAP stability across models")
    parser.add_argument("--dataset", choices=["california", "adult", "titanic"], default="california")
    parser.add_argument("--data-path", default=None, help="Path to local CSV (for california: must have MedHouseVal column)")
    parser.add_argument("--shap-sample", type=int, default=200, help="Rows to use for SHAP (background + explain)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k features for overlap metric")
    args = parser.parse_args()

    print("Loading data...")
    X, y, feature_names, task = load_data(args.dataset, data_path=args.data_path)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Subsample for SHAP
    n_sample = min(args.shap_sample, len(X_test))
    X_explain = X_test.iloc[:n_sample]
    X_background = X_train.iloc[:min(100, len(X_train))]

    models = get_models(task)
    print(f"Training {list(models.keys())}...")

    importance_ranks = {}
    mean_abs_shap_by_model = {}
    metrics_by_model = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        if task == "regression":
            y_pred = model.predict(X_test)
            metrics_by_model[name] = {"RMSE": np.sqrt(mean_squared_error(y_test, y_pred)), "R2": r2_score(y_test, y_pred)}
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            metrics_by_model[name] = {"Accuracy": accuracy_score(y_test, y_pred), "AUC": roc_auc_score(y_test, y_proba)}
        explainer = shap.TreeExplainer(model, X_background)
        mean_abs = mean_abs_shap_importance(explainer, X_background, X_explain)
        mean_abs_shap_by_model[name] = mean_abs
        importance_ranks[name] = np.argsort(np.argsort(-mean_abs))

    print("\n--- Model performance (test set) ---")
    for name in models:
        m = metrics_by_model[name]
        parts = [f"{k}: {v:.4f}" for k, v in m.items()]
        print(f"  {name}: {', '.join(parts)}")

    # Stability metrics
    print("\n--- Pairwise Spearman (feature rank correlation) ---")
    for m1, m2, r, p in spearman_rank_correlation(importance_ranks):
        print(f"  {m1} vs {m2}: r = {r:.3f}, p = {p:.4f}")

    print(f"\n--- Top-{args.top_k} feature overlap (Jaccard) ---")
    for m1, m2, j in top_k_overlap(importance_ranks, k=args.top_k):
        print(f"  {m1} vs {m2}: {j:.3f}")

    # Top features per model
    print("\n--- Top 5 features by mean |SHAP| per model ---")
    for name in models:
        order = np.argsort(-mean_abs_shap_by_model[name])
        top = [feature_names[i] for i in order[:5]]
        print(f"  {name}: {top}")


if __name__ == "__main__":
    main()
