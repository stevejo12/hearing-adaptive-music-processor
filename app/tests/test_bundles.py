import io
import json
import zipfile

import pytest

from cadenza_app import models


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


def make_zip(path, files):
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return path


GOOD = {"Flute/config.json": json.dumps({"a": 1}), "Flute/model.safetensors": b"w" * 100,
        "Oboe/config.json": "{}", "Oboe/model.safetensors": b"w" * 50, "README.txt": "hi"}


def test_register_good_bundle(tmp_path):
    seen = []
    meta = models.register_bundle(make_zip(tmp_path / "m.zip", GOOD), "  My   fine-tune ", 10_000,
                                  check=lambda p: seen.append(p))
    assert meta["name"] == "My fine-tune" and meta["instruments"] == ["Oboe", "Flute"] or \
        set(meta["instruments"]) == {"Flute", "Oboe"}
    assert len(seen) == 2
    listed = {m["id"]: m for m in models.list_models()}
    assert meta["id"] in listed and "baseline" in listed
    assert (tmp_path / "models" / meta["id"] / "Flute" / "model.safetensors").read_bytes() == b"w" * 100


def test_wrapping_folder_is_stripped(tmp_path):
    files = {f"my_model/{k}": v for k, v in GOOD.items()}
    meta = models.register_bundle(make_zip(tmp_path / "m.zip", files), "x", 10_000)
    assert set(meta["instruments"]) == {"Flute", "Oboe"}


def test_rejects_pickle_weights(tmp_path):
    files = dict(GOOD, **{"Flute/model.pt": b"pickle"})
    with pytest.raises(ValueError, match="Pickle"):
        models.register_bundle(make_zip(tmp_path / "m.zip", files), "x", 10_000)


@pytest.mark.parametrize("evil", ["../evil.txt", "/abs/evil.txt", "a\\..\\evil.txt"])
def test_rejects_unsafe_paths(tmp_path, evil):
    with pytest.raises(ValueError, match="unsafe"):
        models.register_bundle(make_zip(tmp_path / "m.zip", dict(GOOD, **{evil: "x"})), "x", 10_000)
    assert not (tmp_path / "evil.txt").exists()


def test_rejects_wrong_layout_not_zip_and_size(tmp_path):
    with pytest.raises(ValueError, match="No usable model"):
        models.register_bundle(make_zip(tmp_path / "m.zip", {"Tuba/config.json": "{}", "Tuba/model.safetensors": b"x"}), "x", 10_000)
    (tmp_path / "n.zip").write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="zip"):
        models.register_bundle(tmp_path / "n.zip", "x", 10_000)
    with pytest.raises(ValueError, match="larger"):
        models.register_bundle(make_zip(tmp_path / "m.zip", GOOD), "x", 100)
    assert not [d for d in (tmp_path / "models").iterdir()] if (tmp_path / "models").exists() else True


def test_failed_check_leaves_nothing_behind(tmp_path):
    def bad(path):
        raise ValueError("does not load")

    with pytest.raises(ValueError, match="does not load"):
        models.register_bundle(make_zip(tmp_path / "m.zip", GOOD), "x", 10_000, check=bad)
    assert list((tmp_path / "models").iterdir()) == []


def test_invalid_config_json(tmp_path):
    files = dict(GOOD, **{"Flute/config.json": "{not json"})
    with pytest.raises(ValueError, match="valid JSON"):
        models.register_bundle(make_zip(tmp_path / "m.zip", files), "x", 10_000)


def test_delete_model(tmp_path):
    meta = models.register_bundle(make_zip(tmp_path / "m.zip", GOOD), "x", 10_000)
    assert models.delete_model(meta["id"]) and models.get_model(meta["id"]) is None
    assert not models.delete_model("baseline") and not models.delete_model("../..")


def test_valid_id():
    assert models.valid_id("baseline") and models.valid_id("0a1b2c3d")
    assert not models.valid_id("../etc") and not models.valid_id("ZZZZZZZZ")
