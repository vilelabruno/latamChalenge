"""Demonstrate the image-to-polygon pipeline on a real GOES-style frame.

Reads one raster from data/img/, applies the same CV steps used in the
project (grayscale, morphological closing, inversion, threshold, contour
extraction), and exports intermediate panels for the paper.
"""
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    fig_dir = root / "article" / "flighttime" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    img_path = sorted((root / "data" / "img").glob("2022-06-01 17:00:00.jpg"))[0]
    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"Could not read image at {img_path}")

    # Optional crop to remove borders and legend; keeps central weather field
    image_bgr = image_bgr[200:-200, 200:-200]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # Morphological closing to fill gaps, then invert (bright cells)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    inv = cv2.normalize(255 - closed, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold and contour extraction
    _, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < 200:  # filter speckles
            continue
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        polygons.append(approx[:, 0, :])

    # Build montage
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].imshow(image_rgb)
    axes[0].set_title("Raw satellite frame (data/img)")

    axes[1].imshow(inv, cmap="gray")
    axes[1].contour(thresh, levels=[0.5], colors="cyan", linewidths=1.0)
    axes[1].set_title("Closing + inversion + threshold")

    axes[2].imshow(image_rgb, alpha=0.6)
    for poly in polygons:
        poly_closed = np.vstack([poly, poly[0]])  # close ring
        axes[2].plot(poly_closed[:, 0], poly_closed[:, 1], color="red", linewidth=1.8)
    axes[2].set_title("Extracted polygons")

    fig.tight_layout()
    fig.savefig(fig_dir / "polygon_pipeline.png", dpi=220, bbox_inches="tight")

    # Overlay-only view for reuse
    plt.figure(figsize=(5, 4))
    plt.imshow(image_rgb, alpha=0.55)
    for poly in polygons:
        poly_closed = np.vstack([poly, poly[0]])
        plt.plot(poly_closed[:, 0], poly_closed[:, 1], color="red", linewidth=1.8)
    plt.xticks([])
    plt.yticks([])
    plt.title("Polygon overlay on real frame")
    plt.tight_layout()
    plt.savefig(fig_dir / "polygon_overlay.png", dpi=220, bbox_inches="tight")

    # Extract a real CAT-62 trajectory on the same day
    target_date = pd.to_datetime("2022-06-01").date()
    track = None
    fid = None
    for chunk in pd.read_csv(
        root / "data" / "test_cat62.csv",
        chunksize=200000,
        parse_dates=["dt_radar"],
    ):
        mask = (
            (chunk["dt_radar"].dt.date == target_date)
            & chunk["lat"].notna()
            & chunk["lon"].notna()
        )
        if mask.any():
            sub = chunk[mask]
            fid = sub.iloc[0]["flightid"]
            track = sub[sub["flightid"] == fid][["dt_radar", "lat", "lon"]].sort_values(
                "dt_radar"
            )
            break

    if track is not None:
        plt.figure(figsize=(5.5, 4.2))
        plt.plot(track["lon"], track["lat"], "-o", ms=3, lw=1.5, color="tab:blue")
        plt.scatter(track["lon"].iloc[0], track["lat"].iloc[0], color="green", s=40, label="start")
        plt.scatter(track["lon"].iloc[-1], track["lat"].iloc[-1], color="red", s=40, label="end")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.title(f"CAT-62 track on 2022-06-01 (flight {fid[:6]}…)")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(fig_dir / "real_trajectory.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
