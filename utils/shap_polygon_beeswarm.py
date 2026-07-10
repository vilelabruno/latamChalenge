#!/usr/bin/env python3
"""
Train a LightGBM regressor on the challenge training matrix and plot a
beeswarm-style SHAP visualization for the convective polygon feature
(`sum(flight_through_area)`).

Outputs: saves a PNG to the path given by --output (default:
article/flighttime/figures/shap_polygon_beeswarm.png).
"""

import argparse
import os
from collections import defaultdict

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    id_cols = {"flightid", "HoraData", "HoraDataDest", "flightid_"}
    target_col = "target"
    feature_cols = [c for c in df.columns if c not in id_cols | {target_col}]
    X = df[feature_cols]
    y = df[target_col]
    return X, y, feature_cols


def fit_model(X: pd.DataFrame, y: pd.Series, seed: int = 42) -> lgb.LGBMRegressor:
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


def beeswarm_positions(values: np.ndarray, bin_width: float = 0.05, step: float = 0.12):
    """
    Lightweight beeswarm layout: bin x-values, then stack points within each bin
    alternating sign to reduce overlap.
    """
    bins = np.floor(values / bin_width)
    stack_counter = defaultdict(int)
    y_positions = np.empty_like(values, dtype=float)
    for i, b in enumerate(bins):
        count = stack_counter[b]
        # Alternate up/down stacking for readability
        y_positions[i] = (count // 2 + 1) * step * (1 if count % 2 else -1)
        stack_counter[b] += 1
    return y_positions


def plot_beeswarm(shap_vals, feature_vals, output_path: str):
    order = np.argsort(shap_vals)
    shap_sorted = shap_vals[order]
    feat_sorted = feature_vals[order]
    y_pos = beeswarm_positions(shap_sorted)

    plt.figure(figsize=(7, 4))
    sc = plt.scatter(
        shap_sorted,
        y_pos,
        c=feat_sorted,
        cmap="viridis",
        s=10,
        alpha=0.6,
        edgecolors="none",
    )
    plt.axvline(0, color="k", linestyle="--", linewidth=0.8)
    plt.xlabel("SHAP contribution (seconds)")
    plt.ylabel("Jittered density")
    plt.title("Beeswarm: polygon exposure vs. SHAP contribution")
    cbar = plt.colorbar(sc)
    cbar.set_label("sum(flight_through_area)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compute SHAP contributions for polygon feature and plot beeswarm."
    )
    parser.add_argument(
        "--data",
        default="data/train_with_flight_v2_greather_1500.csv",
        help="Path to the training matrix CSV.",
    )
    parser.add_argument(
        "--output",
        default="article/flighttime/figures/shap_polygon_beeswarm.png",
        help="Output path for the beeswarm PNG.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for split/model."
    )
    args = parser.parse_args()

    X, y, feature_cols = load_data(args.data)
    polygon_feat = "sum(flight_through_area)"
    if polygon_feat not in feature_cols:
        raise SystemExit(f"{polygon_feat} not found in features.")

    model = fit_model(X, y, seed=args.seed)
    # Sample for visualization to keep plot size manageable
    sample_size = min(8000, len(X))
    rs = np.random.RandomState(args.seed)
    sample_idx = rs.choice(len(X), size=sample_size, replace=False)
    X_sample = X.iloc[sample_idx].copy()

    shap_matrix = model.predict(X_sample, pred_contrib=True)
    poly_idx = feature_cols.index(polygon_feat)
    shap_polygon = shap_matrix[:, poly_idx]
    poly_vals = X_sample[polygon_feat].fillna(0).values

    plot_beeswarm(shap_polygon, poly_vals, args.output)

    mean_abs_shap = float(np.mean(np.abs(shap_polygon)))
    p95_poly = float(np.nanpercentile(poly_vals, 95))
    print(f"Saved beeswarm to {args.output}")
    print(f"Mean |SHAP| for polygon: {mean_abs_shap:.2f} s")
    print(f"95th percentile polygon exposure: {p95_poly:.3f}")


if __name__ == "__main__":
    main()
