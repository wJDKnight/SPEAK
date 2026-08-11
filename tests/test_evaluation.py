import numpy as np
import pandas as pd
import pytest

from src.evaluation import asw, chaos, evaluate_dataframe, pas


def test_perfect_labels_have_perfect_accuracy_metrics():
    frame = pd.DataFrame(
        {
            "truth": ["a", "a", "a", "b", "b", "b"],
            "prediction": ["a", "a", "a", "b", "b", "b"],
            "x": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    )
    metrics = evaluate_dataframe(frame, "truth", "prediction")
    for name in ("ARI", "NMI", "HOM", "COM"):
        assert metrics[name] == pytest.approx(1.0)
    assert metrics["CHAOS"] >= 0
    assert 0 <= metrics["PAS"] <= 1
    assert -1 <= metrics["ASW"] <= 1


def test_spatial_metrics_reject_mismatched_rows():
    with pytest.raises(ValueError, match="same number"):
        chaos(["a", "b"], np.zeros((3, 2)))


def test_asw_rejects_one_cluster():
    with pytest.raises(ValueError, match="ASW requires"):
        asw(["a", "a", "a"], np.array([[0, 0], [1, 0], [2, 0]]))


def test_pas_handles_fewer_than_ten_neighbours():
    labels = ["a", "a", "b", "b"]
    coordinates = np.array([[0, 0], [0, 1], [10, 0], [10, 1]])
    value = pas(labels, coordinates)
    assert 0 <= value <= 1
