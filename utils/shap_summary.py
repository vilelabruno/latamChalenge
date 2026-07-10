#!/usr/bin/env python3
"""
Train a LightGBM regressor on the training matrix and generate
SHAP-style beeswarm plots:
  1) Top-k features by mean |SHAP|, colored by feature value.
  2) Detailed beeswarm for the polygon exposure feature.

Outputs:
  - article/flighttime/figures/shap_summary.png
  - article/flighttime/figures/shap_polygon_beeswarm.png
"""

import argparse
import os
from typing import List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    df = pd.read_csv(csv_path)
    id_cols = {"flightid", "HoraData", "HoraDataDest", "flightid_"}
    target_col = "target"
    feature_cols = [c for c in df.columns if c not in id_cols | {target_col}]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def fit_model(X: pd.DataFrame, y: pd.Series, seed: int) -> lgb.LGBMRegressor:
    params = dict(
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        feature_fraction=0.9,
        bagging_fraction=0.8,
        bagging_freq=5,
        random_state=seed,
        objective="rmse",
    )
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model


def normalize(vals: np.ndarray) -> np.ndarray:
    """Normalize to [0, 1] using percentiles to reduce outlier impact."""
    lower, upper = np.percentile(vals, [1, 99])
    clipped = np.clip(vals, lower, upper)
    return (clipped - lower) / (upper - lower + 1e-9)


def plot_summary(
    shap_matrix: np.ndarray,
    X_sample: pd.DataFrame,
    feature_cols: List[str],
    output_path: str,
    top_k: int = 15,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    # Discard last column (base value) from pred_contrib output
    shap_feat = shap_matrix[:, :-1]
    mean_abs = np.mean(np.abs(shap_feat), axis=0)
    top_idx = np.argsort(mean_abs)[::-1][:top_k]
    top_features = [feature_cols[i] for i in top_idx]

    plt.figure(figsize=(8, 6))
    for y_idx, feat_name in enumerate(top_features):
        col_idx = feature_cols.index(feat_name)
        shap_vals = shap_feat[:, col_idx]
        feat_vals = X_sample.iloc[:, col_idx].values
        colors = normalize(feat_vals)

        jitter = rng.uniform(-0.35, 0.35, size=shap_vals.shape)
        plt.scatter(
            shap_vals,
            np.full_like(shap_vals, y_idx) + jitter,
            c=colors,
            cmap="coolwarm",
            s=8,
            alpha=0.6,
            edgecolors="none",
        )

    plt.yticks(range(len(top_features)), top_features)
    plt.axvline(0, color="k", linestyle="--", linewidth=0.8)
    cbar = plt.colorbar()
    cbar.set_label("Feature value (normalized)")
    plt.xlabel("SHAP contribution (seconds)")
    plt.title(f"SHAP beeswarm (top {top_k} features)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    return top_features, mean_abs[top_idx]


def plot_polygon_beeswarm(
    shap_matrix: np.ndarray,
    X_sample: pd.DataFrame,
    feature_cols: List[str],
    output_path: str,
    seed: int = 42,
):
    polygon_feat = "sum(flight_through_area)"
    if polygon_feat not in feature_cols:
        raise SystemExit(f"{polygon_feat} not found in features.")
    poly_idx = feature_cols.index(polygon_feat)
    shap_feat = shap_matrix[:, :-1]
    shap_vals = shap_feat[:, poly_idx]
    feat_vals = X_sample[polygon_feat].fillna(0).values

    order = np.argsort(shap_vals)
    shap_sorted = shap_vals[order]
    feat_sorted = feat_vals[order]

    # simple beeswarm stacking
    bin_width = 0.05
    bins = np.floor(shap_sorted / bin_width)
    counts = {}
    y_positions = np.empty_like(shap_sorted, dtype=float)
    for i, b in enumerate(bins):
        c = counts.get(b, 0)
        y_positions[i] = (c // 2 + 1) * 0.12 * (1 if c % 2 else -1)
        counts[b] = c + 1

    colors = normalize(feat_sorted)
    plt.figure(figsize=(7, 4))
    sc = plt.scatter(
        shap_sorted,
        y_positions,
        c=colors,
        cmap="coolwarm",
        s=10,
        alpha=0.6,
        edgecolors="none",
    )
    plt.axvline(0, color="k", linestyle="--", linewidth=0.8)
    plt.xlabel("SHAP contribution (seconds)")
    plt.ylabel("Jittered density")
    plt.title("Polygon exposure vs. SHAP contribution")
    cbar = plt.colorbar(sc)
    cbar.set_label("sum(flight_through_area) (normalized)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP-style beeswarm plots for the model."
    )
    parser.add_argument(
        "--data",
        default="data/train_with_flight_v2_greather_1500.csv",
        help="Path to training CSV.",
    )
    parser.add_argument(
        "--summary-output",
        default="article/flighttime/figures/shap_summary.png",
        help="Output path for the top-features beeswarm.",
    )
    parser.add_argument(
        "--importance-output",
        default="article/flighttime/figures/shap_importance.png",
        help="Output path for the mean |SHAP| bar plot.",
    )
    parser.add_argument(
        "--polygon-output",
        default="article/flighttime/figures/shap_polygon_beeswarm.png",
        help="Output path for the polygon beeswarm.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--sample",
        type=int,
        default=8000,
        help="Sample size for SHAP plotting (controls runtime/size).",
    )
    args = parser.parse_args()

    X, y, feature_cols = load_data(args.data)
    model = fit_model(X, y, seed=args.seed)

    sample_size = min(args.sample, len(X))
    rs = np.random.RandomState(args.seed)
    sample_idx = rs.choice(len(X), size=sample_size, replace=False)
    X_sample = X.iloc[sample_idx].copy()

    shap_matrix = model.predict(X_sample, pred_contrib=True)

    top_feats, top_means = plot_summary(
        shap_matrix, X_sample, feature_cols, args.summary_output, seed=args.seed
    )
    # Bar plot for the same top-k features
    plt.figure(figsize=(6, 4))
    y_pos = np.arange(len(top_feats))[::-1]
    plt.barh(y_pos, top_means[::-1], color="steelblue")
    plt.yticks(y_pos, top_feats[::-1])
    plt.xlabel("Mean |SHAP| (seconds)")
    plt.title("Top features by mean |SHAP|")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.importance_output), exist_ok=True)
    plt.savefig(args.importance_output, dpi=200)
    plt.close()

    plot_polygon_beeswarm(
        shap_matrix, X_sample, feature_cols, args.polygon_output, seed=args.seed
    )

    print(f"Saved summary beeswarm to {args.summary_output}")
    print(f"Saved importance bars to {args.importance_output}")
    print(f"Saved polygon beeswarm to {args.polygon_output}")
    print("Top features by mean |SHAP|:")
    for name, val in zip(top_feats, top_means):
        print(f"  {name}: {val:.2f} s")


if __name__ == "__main__":
    main()
