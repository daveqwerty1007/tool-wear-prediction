import os
import sys
import pandas as pd
import pytest

# Ensure the src package is importable when running `pytest` directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_loader import load_vicomtech_data


def test_file_not_found(tmp_path):
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        load_vicomtech_data(raw_path=str(missing))


def test_load_all_columns(tmp_path):
    # create a small CSV with extra spaces in column names
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "Tool ": [1, 2],
        "AE_RMS": [0.1, 0.2],
        "AE_MAX": [1, 2],
        "F_c_RMS": [3, 4],
        "F_c_MAX": [5, 6],
        "Vb": [0.01, 0.02],
        "Extra": [7, 8]
    })
    df.to_csv(csv_path, index=False)

    loaded = load_vicomtech_data(raw_path=str(csv_path), selected_only=False)
    # expect columns to be renamed and the extra column preserved
    expected = {
        "tool_id",
        "acoustic_rms",
        "acoustic_peak",
        "cutting_force_rms",
        "cutting_force_max",
        "flank_wear",
        "extra"
    }
    assert expected.issubset(loaded.columns)
    # engineered columns should be present
    for col in ["force_ratio", "force_diff", "acoustic_mean", "acoustic_diff", "acoustic_to_force"]:
        assert col in loaded.columns
    assert len(loaded) == 2


def test_selected_only_subset(tmp_path):
    csv_path = tmp_path / "data.csv"
    df = pd.DataFrame({
        "Tool": [1, 2],
        "AE_RMS": [0.1, 0.2],
        "AE_MAX": [1, 2],
        "F_c_RMS": [3, 4],
        "F_c_MAX": [5, 6],
        "Vb": [0.01, 0.02],
        "Extra": [7, 8]
    })
    df.to_csv(csv_path, index=False)

    loaded = load_vicomtech_data(raw_path=str(csv_path), selected_only=True)
    expected = {
        "tool_id",
        "acoustic_rms",
        "acoustic_peak",
        "cutting_force_rms",
        "cutting_force_max",
        "force_ratio",
        "force_diff",
        "acoustic_mean",
        "acoustic_diff",
        "acoustic_to_force",
        "flank_wear"
    }
    assert set(loaded.columns) == expected

