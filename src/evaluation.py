"""Unified evaluation metrics for spatial-domain annotations.

The spatial continuity definitions follow the SDMBench formulations used by
the original analysis: standardized-coordinate 1-NN CHAOS, 10-NN PAS, and
Euclidean average silhouette width (ASW).
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


METRIC_COLUMNS = ("ARI", "NMI", "HOM", "COM", "CHAOS", "PAS", "ASW")


def _arrays(labels: Iterable[object], coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    label_array = np.asarray(list(labels), dtype=object)
    coordinate_array = np.asarray(coordinates, dtype=float)
    if coordinate_array.ndim != 2 or coordinate_array.shape[1] < 2:
        raise ValueError("coordinates must be an n-by-2 (or wider) array")
    if len(label_array) != len(coordinate_array):
        raise ValueError("labels and coordinates must contain the same number of rows")
    if len(label_array) < 3:
        raise ValueError("at least three observations are required")
    if pd.isna(label_array).any() or not np.isfinite(coordinate_array).all():
        raise ValueError("labels and coordinates must not contain missing values")
    return label_array, coordinate_array


def chaos(labels: Iterable[object], coordinates: np.ndarray) -> float:
    """Mean within-domain nearest-neighbour distance after coordinate scaling."""

    label_array, coordinate_array = _arrays(labels, coordinates)
    scaled = StandardScaler().fit_transform(coordinate_array)
    total = 0.0
    for label in np.unique(label_array):
        cluster = scaled[label_array == label]
        if len(cluster) <= 2:
            continue
        distances, _ = NearestNeighbors(n_neighbors=2).fit(cluster).kneighbors(cluster)
        total += float(distances[:, 1].sum())
    return total / len(label_array)


def pas(labels: Iterable[object], coordinates: np.ndarray, k: int = 10) -> float:
    """Fraction of observations whose local-neighbour majority has another label."""

    label_array, coordinate_array = _arrays(labels, coordinates)
    if k < 1:
        raise ValueError("k must be positive")
    effective_k = min(k, len(label_array) - 1)
    indices = NearestNeighbors(n_neighbors=effective_k + 1).fit(coordinate_array).kneighbors(
        coordinate_array, return_distance=False
    )[:, 1:]
    mismatches = label_array[indices] != label_array[:, None]
    return float((mismatches.sum(axis=1) > (effective_k / 2)).mean())


def asw(labels: Iterable[object], coordinates: np.ndarray) -> float:
    """Euclidean average silhouette width for predicted spatial domains."""

    label_array, coordinate_array = _arrays(labels, coordinates)
    n_labels = len(np.unique(label_array))
    if n_labels < 2 or n_labels >= len(label_array):
        raise ValueError("ASW requires between 2 and n-1 unique labels")
    return float(silhouette_score(coordinate_array, label_array, metric="euclidean"))


def evaluate_dataframe(
    frame: pd.DataFrame,
    truth_column: str,
    prediction_column: str,
    x_column: str = "x",
    y_column: str = "y",
) -> dict[str, float]:
    """Calculate all accuracy and spatial-continuity metrics for a result table."""

    required = [truth_column, prediction_column, x_column, y_column]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    clean = frame.loc[:, required].dropna()
    if len(clean) != len(frame):
        raise ValueError("metric input contains missing truth, prediction, or coordinate values")

    truth = clean[truth_column].astype(str).to_numpy()
    prediction = clean[prediction_column].astype(str).to_numpy()
    coordinates = clean[[x_column, y_column]].to_numpy(dtype=float)
    return {
        "ARI": float(adjusted_rand_score(truth, prediction)),
        "NMI": float(normalized_mutual_info_score(truth, prediction)),
        "HOM": float(homogeneity_score(truth, prediction)),
        "COM": float(completeness_score(truth, prediction)),
        "CHAOS": chaos(prediction, coordinates),
        "PAS": pas(prediction, coordinates),
        "ASW": asw(prediction, coordinates),
    }
