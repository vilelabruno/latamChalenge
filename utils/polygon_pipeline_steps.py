"""Export step-by-step imagery for the polygon pipeline and a route overlay."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath


def load_polygons_for_time(csv_path: Path, timestamp: str) -> list[np.ndarray]:
    rows = []
    for chunk in pd.read_csv(
        csv_path,
        usecols=["data", "poly", "lat", "lon", "vertice"],
        chunksize=200000,
    ):
        sub = chunk[chunk["data"] == timestamp]
        if not sub.empty:
            rows.append(sub)
    if not rows:
        raise ValueError(f"No polygons found for {timestamp}")
    df = pd.concat(rows, ignore_index=True)
    polygons = []
    for _, grp in df.groupby("poly"):
        grp = grp.sort_values("vertice")
        coords = np.column_stack([grp["lon"].values, grp["lat"].values])
        if len(coords) >= 3:
            polygons.append(coords)
    return polygons


def load_cat62_hour(cat62_path: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    points = []
    for chunk in pd.read_csv(
        cat62_path,
        usecols=["flightid", "lat", "lon", "dt_radar"],
        chunksize=500000,
    ):
        dt = pd.to_datetime(chunk["dt_radar"], unit="ms", errors="coerce", utc=True)
        mask = (dt >= start) & (dt < end)
        if mask.any():
            sub = chunk.loc[mask, ["flightid", "lat", "lon"]].copy()
            sub["dt_radar"] = dt[mask].values
            points.append(sub)
    if not points:
        raise ValueError("No CAT-62 points found in the requested window.")
    return pd.concat(points, ignore_index=True)


def pick_flight_with_intersections(polygons: list[np.ndarray], points_df: pd.DataFrame):
    bboxes = []
    paths = []
    for coords in polygons:
        minx = coords[:, 0].min()
        maxx = coords[:, 0].max()
        miny = coords[:, 1].min()
        maxy = coords[:, 1].max()
        bboxes.append((minx, miny, maxx, maxy))
        paths.append(MplPath(coords))

    pts = points_df[["lon", "lat"]].values
    inside_any = np.zeros(len(pts), dtype=bool)
    for bbox, path in zip(bboxes, paths):
        minx, miny, maxx, maxy = bbox
        mask = (
            (pts[:, 0] >= minx)
            & (pts[:, 0] <= maxx)
            & (pts[:, 1] >= miny)
            & (pts[:, 1] <= maxy)
        )
        if not mask.any():
            continue
        inside = path.contains_points(pts[mask])
        if inside.any():
            inside_any[mask] |= inside

    points_df = points_df.copy()
    points_df["inside"] = inside_any
    hits = points_df.groupby("flightid")["inside"].sum().sort_values(ascending=False)
    if hits.empty or hits.iloc[0] == 0:
        # Fall back to the densest track in the window.
        best_fid = points_df["flightid"].value_counts().idxmax()
    else:
        best_fid = hits.index[0]
    return best_fid, points_df


def export_pipeline_steps(img_path: Path, fig_dir: Path):
    image_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise SystemExit(f"Could not read image at {img_path}")

    # Crop borders/legend to keep the main weather field.
    image_bgr = image_bgr[200:-200, 200:-200]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closing = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    inversion = (closing.astype(np.int16) * -1) + int(closing.max())
    inversion = np.clip(inversion, 0, 255).astype(np.uint8)
    inversion = cv2.normalize(inversion, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    inversion[inversion < 100] = 0
    _, thresh = cv2.threshold(inversion, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polygons = []
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        polygons.append(approx[:, 0, :])

    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.imsave(fig_dir / "pipeline_raw.png", image_rgb)
    plt.imsave(fig_dir / "pipeline_gray.png", gray, cmap="gray")
    plt.imsave(fig_dir / "pipeline_closing.png", closing, cmap="gray")
    plt.imsave(fig_dir / "pipeline_inversion.png", inversion, cmap="gray")
    plt.imsave(fig_dir / "pipeline_threshold.png", thresh, cmap="gray")

    plt.figure(figsize=(6, 5))
    plt.imshow(image_rgb, alpha=0.7)
    for poly in polygons:
        poly_closed = np.vstack([poly, poly[0]])
        plt.plot(poly_closed[:, 0], poly_closed[:, 1], color="red", linewidth=1.2)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(fig_dir / "pipeline_polygons.png", dpi=220, bbox_inches="tight")
    plt.close("all")


def export_route_overlay(
    polygons: list[np.ndarray],
    points_df: pd.DataFrame,
    flight_id: str,
    fig_dir: Path,
    timestamp: str,
):
    df_flight = points_df[points_df["flightid"] == flight_id].copy()
    if "dt_radar" in df_flight.columns:
        df_flight = df_flight.sort_values("dt_radar")
    else:
        df_flight = df_flight.sort_values(["lon", "lat"])

    # Determine which track points are inside any polygon.
    bboxes = []
    paths = []
    for coords in polygons:
        minx = coords[:, 0].min()
        maxx = coords[:, 0].max()
        miny = coords[:, 1].min()
        maxy = coords[:, 1].max()
        bboxes.append((minx, miny, maxx, maxy))
        paths.append(MplPath(coords))

    pts = df_flight[["lon", "lat"]].values
    inside_any = np.zeros(len(pts), dtype=bool)
    poly_hit = np.zeros(len(polygons), dtype=bool)
    for idx, (bbox, path) in enumerate(zip(bboxes, paths)):
        minx, miny, maxx, maxy = bbox
        mask = (
            (pts[:, 0] >= minx)
            & (pts[:, 0] <= maxx)
            & (pts[:, 1] >= miny)
            & (pts[:, 1] <= maxy)
        )
        if not mask.any():
            continue
        inside = path.contains_points(pts[mask])
        if inside.any():
            inside_any[mask] |= inside
            poly_hit[idx] = True

    df_flight["inside"] = inside_any

    plt.figure(figsize=(6.2, 5.2))
    for idx, poly in enumerate(polygons):
        color = "tab:orange" if poly_hit[idx] else "lightgray"
        poly_closed = np.vstack([poly, poly[0]])
        plt.plot(poly_closed[:, 0], poly_closed[:, 1], color=color, linewidth=0.8)

    plt.plot(df_flight["lon"], df_flight["lat"], color="tab:blue", linewidth=1.4)
    inside_pts = df_flight[df_flight["inside"]]
    if not inside_pts.empty:
        plt.scatter(
            inside_pts["lon"],
            inside_pts["lat"],
            color="crimson",
            s=12,
            label="Inside polygon",
            zorder=3,
        )
    plt.scatter(
        df_flight["lon"].iloc[0],
        df_flight["lat"].iloc[0],
        color="green",
        s=35,
        label="Start",
        zorder=4,
    )
    plt.scatter(
        df_flight["lon"].iloc[-1],
        df_flight["lat"].iloc[-1],
        color="red",
        s=35,
        label="End",
        zorder=4,
    )
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title(f"Route vs. polygons at {timestamp} (flight {flight_id[:6]}…)")
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(fig_dir / "route_polygon_intersection.png", dpi=220, bbox_inches="tight")
    plt.close("all")


def main():
    root = Path(__file__).resolve().parents[1]
    fig_dir = root / "article" / "flighttime" / "figures"

    timestamp = "2022-06-01 17:00:00"
    img_path = root / "data" / "img" / f"{timestamp}.jpg"
    export_pipeline_steps(img_path, fig_dir)

    polygons = load_polygons_for_time(
        root / "data" / "imageDataLatLon.csv",
        f"{timestamp}.jpg",
    )
    start = pd.Timestamp(timestamp, tz="UTC")
    end = start + pd.Timedelta(hours=1)
    points_df = load_cat62_hour(root / "data" / "cat62.csv", start, end)
    flight_id, points_df = pick_flight_with_intersections(polygons, points_df)
    export_route_overlay(polygons, points_df, flight_id, fig_dir, timestamp)


if __name__ == "__main__":
    main()
