import pytest

from deepfake_detector.metadata import describe, load_metadata, split_by_group


def test_load_skips_missing_files_and_groups_fakes_with_their_original(dfdc_folder):
    df = load_metadata(str(dfdc_folder))
    assert len(df) == 24
    assert "missing.mp4" not in set(df["video"])
    fakes = df[df["label"] == 1]
    assert (fakes["group"] == fakes["original"]).all()
    reals = df[df["label"] == 0]
    assert (reals["group"] == reals["video"]).all()


def test_no_group_is_split_across_train_val_and_test(dfdc_folder):
    df = split_by_group(load_metadata(dfdc_folder), 0.25, 0.25, seed=1)
    assert set(df["split"]) == {"train", "val", "test"}
    assert (df.groupby("group")["split"].nunique() == 1).all()


def test_split_is_reproducible(dfdc_folder):
    df = load_metadata(dfdc_folder)
    a = split_by_group(df, 0.25, 0.25, seed=7)["split"].tolist()
    b = split_by_group(df, 0.25, 0.25, seed=7)["split"].tolist()
    assert a == b


def test_describe_counts_real_and_fake(dfdc_folder):
    table = describe(split_by_group(load_metadata(dfdc_folder), 0.25, 0.25, seed=1))
    assert table[["real", "fake"]].to_numpy().sum() == 24


def test_bad_fractions_are_rejected(dfdc_folder):
    with pytest.raises(ValueError):
        split_by_group(load_metadata(dfdc_folder), 0.5, 0.5, seed=1)


def test_folder_without_metadata_fails_clearly(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_metadata(str(tmp_path))
