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


def local_stability_metrics(shap_per_model, top_k=5):
    """
    Per-instance stability: for each instance, compute pairwise Spearman and top-k
    overlap of feature ranks (by |SHAP|), then summarize.
    shap_per_model: dict of model_name -> array (n_instances, n_features)
    Returns: dict with mean_spearman, std_spearman, prop_stable_spearman (r>=0.9),
             mean_topk_jaccard, prop_stable_topk (jaccard=1 for all pairs).
    """
    from scipy.stats import spearmanr
    names = list(shap_per_model.keys())
    n_inst, n_feat = shap_per_model[names[0]].shape
    spearman_per_inst = []
    topk_jaccard_per_inst = []
    for idx in range(n_inst):
        ranks = {}
        for name in names:
            abs_shap = np.abs(shap_per_model[name][idx])
            ranks[name] = np.argsort(np.argsort(-abs_shap))
        # Mean pairwise Spearman for this instance
        pair_rs = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r, _ = spearmanr(ranks[names[i]], ranks[names[j]])
                pair_rs.append(r if not np.isnan(r) else 0.0)
        spearman_per_inst.append(np.mean(pair_rs))
        # Min top-k Jaccard across pairs for this instance (1 = all pairs agree on top-k)
        pair_j = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                top_i = set(np.argsort(ranks[names[i]])[:top_k])
                top_j = set(np.argsort(ranks[names[j]])[:top_k])
                jaccard = len(top_i & top_j) / len(top_i | top_j) if (top_i | top_j) else 1.0
                pair_j.append(jaccard)
        topk_jaccard_per_inst.append(np.min(pair_j))
    spearman_per_inst = np.array(spearman_per_inst)
    topk_jaccard_per_inst = np.array(topk_jaccard_per_inst)
    return {
        "mean_spearman": float(np.mean(spearman_per_inst)),
        "std_spearman": float(np.std(spearman_per_inst)),
        "prop_stable_spearman": float(np.mean(spearman_per_inst >= 0.9)),
        "mean_topk_jaccard": float(np.mean(topk_jaccard_per_inst)),
        "prop_stable_topk": float(np.mean(topk_jaccard_per_inst >= 1.0)),
    }


def _update_local_stability_tex():
    """Update paper/main.tex local stability table from all paper/exported_<dataset>.json."""
    import json
    import re
    tex_path = "paper/main.tex"
    if not os.path.isfile(tex_path):
        return
    datasets = ["california", "adult", "titanic"]
    display = {"california": "California", "adult": "Adult", "titanic": "Titanic"}
    rows_data = []
    shap_sample = None
    for d in datasets:
        path = f"paper/exported_{d}.json"
        if os.path.isfile(path):
            with open(path) as f:
                data = json.load(f)
            ls = data.get("local_stability", {})
            if shap_sample is None:
                shap_sample = data.get("shap_sample", 200)
            rows_data.append((
                display[d],
                ls.get("mean_spearman"), ls.get("std_spearman"),
                ls.get("prop_stable_spearman"), ls.get("mean_topk_jaccard"), ls.get("prop_stable_topk"),
            ))
        else:
            rows_data.append((display[d], None, None, None, None, None))
    with open(tex_path) as f:
        tex = f.read()
    # Build new table rows
    lines = []
    for label, m, s, p_s, j, p_t in rows_data:
        if m is not None:
            pct_s = int(round(p_s * 100))
            pct_t = int(round(p_t * 100))
            lines.append(f"    {label:10} & {m:.2f} ({s:.2f}) & {pct_s}\\%  & {j:.2f} & {pct_t}\\% \\\\")
        else:
            lines.append(f"    {label:10} & -- (--) & --\\%  & -- & --\\% \\\\")
    new_block = "\n".join(lines)
    # Replace caption instance count (first occurrence in caption for tab:local)
    if shap_sample is not None:
        tex = re.sub(
            r"(single run, )\d+( instances\))",
            rf"\g<1>{shap_sample}\2",
            tex,
            count=1,
        )
    # Replace the three data rows (between \midrule and \bottomrule in tab:local)
    pattern = r"(\\label\{tab:local\}\s+\\begin\{tabular\}\{lcccc\}\s+\\toprule\s+Dataset.*?\\midrule\s+)(.*?)(\s+\\bottomrule\s+\\end\{tabular\})"
    match = re.search(pattern, tex, re.DOTALL)
    if match:
        tex = tex[: match.start(2)] + new_block + "\n" + match.group(3) + tex[match.end(3) :]
    with open(tex_path, "w") as f:
        f.write(tex)
    print(f"Updated {tex_path} local stability table (n={shap_sample})")


def main():
    parser = argparse.ArgumentParser(description="SHAP stability across models")
    parser.add_argument("--dataset", choices=["california", "adult", "titanic"], default="california")
    parser.add_argument("--data-path", default=None, help="Path to local CSV (for california: must have MedHouseVal column)")
    parser.add_argument("--shap-sample", type=int, default=200, help="Rows to use for SHAP (background + explain)")
    parser.add_argument("--top-k", type=int, default=5, help="Top-k features for overlap metric")
    parser.add_argument("--save-figures", action="store_true", help="Save importance bar chart to --figures-dir")
    parser.add_argument("--figures-dir", default="paper/figures", help="Directory for saved figures")
    parser.add_argument("--export-tables", action="store_true", help="Write metrics to paper/exported_<dataset>.json for verifying table numbers")
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
    shap_per_model = {}

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
        shap_vals = explainer.shap_values(X_explain, check_additivity=False)
        if isinstance(shap_vals, list):
            shap_vals = np.sum(np.abs(shap_vals), axis=0)
        shap_per_model[name] = np.array(shap_vals)
        mean_abs = np.abs(shap_per_model[name]).mean(axis=0)
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

    # Instance-level (local) stability
    local_metrics = local_stability_metrics(shap_per_model, top_k=args.top_k)
    print("\n--- Instance-level (local) stability ---")
    print(f"  Mean pairwise Spearman per instance: {local_metrics['mean_spearman']:.3f} (std {local_metrics['std_spearman']:.3f})")
    print(f"  Proportion of instances with mean pairwise r >= 0.9: {local_metrics['prop_stable_spearman']:.2%}")
    print(f"  Mean min top-{args.top_k} Jaccard per instance: {local_metrics['mean_topk_jaccard']:.3f}")
    print(f"  Proportion of instances with all pairs top-{args.top_k} identical: {local_metrics['prop_stable_topk']:.2%}")

    # Top features per model
    print("\n--- Top 5 features by mean |SHAP| per model ---")
    top5_by_model = {}
    for name in models:
        order = np.argsort(-mean_abs_shap_by_model[name])
        top = [feature_names[i] for i in order[:5]]
        top5_by_model[name] = top
        print(f"  {name}: {top}")

    if args.export_tables:
        import json
        import os
        os.makedirs("paper", exist_ok=True)
        out = {
            "dataset": args.dataset,
            "shap_sample": args.shap_sample,
            "metrics": {name: metrics_by_model[name] for name in models},
            "spearman": [(m1, m2, float(r)) for m1, m2, r, _ in spearman_rank_correlation(importance_ranks)],
            "top5_overlap": [(m1, m2, float(j)) for m1, m2, j in top_k_overlap(importance_ranks, k=args.top_k)],
            "top5_by_model": top5_by_model,
            "local_stability": {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in local_metrics.items()},
        }
        path = f"paper/exported_{args.dataset}.json"
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Exported table data: {path}")
        _update_local_stability_tex()

    if args.save_figures:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os
        os.makedirs(args.figures_dir, exist_ok=True)
        n_f = len(feature_names)
        x = np.arange(n_f)
        w = 0.25
        fig, ax = plt.subplots(figsize=(max(6, n_f * 0.5), 4))
        for i, (name, mean_abs) in enumerate(mean_abs_shap_by_model.items()):
            ax.bar(x + i * w, mean_abs, w, label=name)
        ax.set_xticks(x + w)
        ax.set_xticklabels(feature_names, rotation=45, ha="right")
        ax.set_ylabel("Mean |SHAP|")
        ax.set_title(f"Global feature importance ({args.dataset})")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(args.figures_dir, f"{args.dataset}_importance.pdf")
        fig.savefig(out)
        plt.close()
        print(f"Saved figure: {out}")


if __name__ == "__main__":
    main()
