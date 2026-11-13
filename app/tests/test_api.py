import io
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

AUD = ",".join(["30"] * 8)


calls = []


@pytest.fixture()
def client(tmp_path, monkeypatch):
    calls.clear()
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    import importlib
    from cadenza_app import db, main, pipeline
    importlib.reload(db)
    importlib.reload(main)

    def fake_process(before, after, instruments, gains, levels_l, levels_r, model_id="baseline", residual=0.1):
        calls.append((model_id, residual))
        after.parent.mkdir(parents=True, exist_ok=True)
        after.write_bytes(b"mp3-fake")
        return {}

    monkeypatch.setattr(main.pipeline, "process", fake_process)
    monkeypatch.setattr(main.models, "start_in_background", lambda: None)  # no real downloads in tests
    monkeypatch.setattr(main.models, "is_ready", lambda: True)
    monkeypatch.setattr(main.pipeline, "check_loadable", lambda path: None)   # no torch in unit tests
    with TestClient(main.app) as c:
        yield c


def login(c, user="admin", pw="admin"):
    return c.post("/login", data={"username": user, "password": pw}, follow_redirects=False)


def upload(c, **over):
    data = {"audiogram_l": AUD, "audiogram_r": AUD, "instruments": "Flute,Oboe,Bassoon", "gains": ""}
    data.update(over)
    return c.post("/api/process", data=data, files={"song": ("a.mp3", io.BytesIO(b"x" * 100), "audio/mpeg")})


def wait_done(c, n=50):
    for _ in range(n):
        rows = c.get("/api/songs").json()
        if rows and all(r["status"] != "processing" for r in rows):
            return rows
        time.sleep(0.05)
    raise AssertionError("job never finished")


def test_requires_login(client):
    assert client.get("/api/songs").status_code == 401
    assert client.get("/", follow_redirects=False).headers["location"] == "/login"
    assert login(client, pw="nope").headers["location"] == "/login?error=1"


def test_full_flow(client):
    assert login(client).status_code == 303
    r = upload(client)
    assert r.status_code == 200, r.text
    row = wait_done(client)[0]
    assert row["status"] == "done" and row["after_url"] and len(row["gains"]) == 3
    assert client.get(row["before_url"]).content == b"x" * 100
    assert client.get(row["after_url"]).content == b"mp3-fake"

    p = client.patch(f"/api/songs/{row['id']}", json={"rating": 4, "comment": "clear flute"})
    assert p.json()["rating"] == 4 and p.json()["comment"] == "clear flute"
    assert client.patch(f"/api/songs/{row['id']}", json={"rating": 9}).status_code == 422

    assert client.delete(f"/api/songs/{row['id']}").status_code == 200
    assert client.get("/api/songs").json() == []
    assert client.get(row["before_url"]).status_code == 404


def test_validation(client):
    login(client)
    assert upload(client, audiogram_l="1,2,3").status_code == 422
    assert upload(client, instruments="Flute").status_code == 422
    assert upload(client, instruments="Flute,Tuba").status_code == 422
    assert upload(client, gains="1,2").status_code == 422
    bad = client.post("/api/process", data={"audiogram_l": AUD, "audiogram_r": AUD, "instruments": "Flute,Oboe"},
                      files={"song": ("a.txt", io.BytesIO(b"x"), "text/plain")})
    assert bad.status_code == 422


def test_upload_limit(client, monkeypatch):
    from cadenza_app import main
    monkeypatch.setattr(main, "MAX_UPLOAD_MB", 0.00005)
    login(client)
    assert upload(client).status_code == 413
    assert client.get("/api/songs").json() == []


def test_failed_job_is_shown(client, monkeypatch):
    from cadenza_app import main

    def boom(*a, **k):
        raise ValueError("Could not decode the audio file")

    monkeypatch.setattr(main.pipeline, "process", boom)
    login(client)
    upload(client)
    row = wait_done(client)[0]
    assert row["status"] == "error" and "decode" in row["error"] and row["after_url"] is None


def test_blocked_while_models_download(client, monkeypatch):
    from cadenza_app import main
    monkeypatch.setattr(main.models, "is_ready", lambda: False)
    login(client)
    assert upload(client).status_code == 503
    assert client.get("/api/songs").json() == []


def model_zip(files=None):
    import json as _json
    import zipfile as _zip
    buf = io.BytesIO()
    with _zip.ZipFile(buf, "w") as zf:
        for k, v in (files or {"Flute/config.json": "{}", "Flute/model.safetensors": b"w" * 64}).items():
            zf.writestr(k, v)
    buf.seek(0)
    return buf


def test_model_upload_select_and_delete(client):
    login(client)
    assert [m["id"] for m in client.get("/api/models").json()] == ["baseline"]
    r = client.post("/api/models", data={"name": "Fine tune 1"}, files={"file": ("ft.zip", model_zip(), "application/zip")})
    assert r.status_code == 200, r.text
    mid = r.json()["id"]
    assert [m["name"] for m in client.get("/api/models").json()] == ["Cadenza baseline (pretrained)", "Fine tune 1"]

    assert upload(client, model=mid, residual="0.3").status_code == 200
    row = wait_done(client)[0]
    assert row["model_id"] == mid and row["model_name"] == "Fine tune 1" and row["residual"] == 0.3
    assert calls[-1] == (mid, 0.3)

    assert client.delete("/api/models/baseline").status_code == 409
    assert client.delete(f"/api/models/{mid}").status_code == 200
    assert client.delete(f"/api/models/{mid}").status_code == 404
    assert upload(client, model=mid).status_code == 422          # gone
    assert client.get("/api/songs").json() == []                  # its table of songs goes with it


def test_model_upload_errors(client):
    login(client)
    bad = client.post("/api/models", data={}, files={"file": ("x.zip", io.BytesIO(b"nope"), "application/zip")})
    assert bad.status_code == 422 and "zip" in bad.json()["detail"]
    pickle = model_zip({"Flute/model.pt": b"x", "Flute/config.json": "{}", "Flute/model.safetensors": b"x"})
    assert client.post("/api/models", data={}, files={"file": ("x.zip", pickle, "application/zip")}).status_code == 422
    assert upload(client, model="../../x").status_code == 422
    assert upload(client, residual="2").status_code == 422


def test_models_need_login(client):
    assert client.get("/api/models").status_code == 401
    assert client.post("/api/models", files={"file": ("x.zip", io.BytesIO(b"x"))}).status_code == 401


def test_each_model_has_its_own_table(client):
    login(client)
    mid = client.post("/api/models", data={"name": "Second"},
                      files={"file": ("m.zip", model_zip(), "application/zip")}).json()["id"]
    assert upload(client).status_code == 200                       # baseline
    wait_done(client)
    assert upload(client, model=mid).status_code == 200
    assert upload(client, model=mid).status_code == 200
    wait_done(client)

    base = client.get("/api/songs", params={"model": "baseline"}).json()
    second = client.get("/api/songs", params={"model": mid}).json()
    assert len(base) == 1 and len(second) == 2
    assert {s["model_id"] for s in base} == {"baseline"} and {s["model_id"] for s in second} == {mid}
    assert len(client.get("/api/songs").json()) == 3                # no filter = everything
    counts = {m["id"]: m["songs"] for m in client.get("/api/models").json()}
    assert counts == {"baseline": 1, mid: 2}

    # rating/comment belong to the row, so they stay in that model's table
    sid = second[0]["id"]
    client.patch(f"/api/songs/{sid}", json={"rating": 5, "comment": "best"})
    assert client.get("/api/songs", params={"model": "baseline"}).json()[0]["rating"] == 0


def test_deleting_a_model_deletes_its_table(client):
    from cadenza_app import db
    login(client)
    mid = client.post("/api/models", data={"name": "Temp"},
                      files={"file": ("m.zip", model_zip(), "application/zip")}).json()["id"]
    upload(client, model=mid); upload(client)
    wait_done(client)
    gone = [s["id"] for s in client.get("/api/songs", params={"model": mid}).json()]
    r = client.delete(f"/api/models/{mid}")
    assert r.json()["songs_deleted"] == 1
    assert client.get("/api/songs", params={"model": mid}).json() == []
    assert len(client.get("/api/songs", params={"model": "baseline"}).json()) == 1
    assert not (db.SONGS_DIR / str(gone[0])).exists()


def test_after_is_served_as_mp3(client):
    login(client)
    upload(client)
    row = wait_done(client)[0]
    r = client.get(row["after_url"])
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/mpeg")


def test_row_carries_all_chosen_settings(client):
    login(client)
    assert upload(client, audiogram_l="25,30,40,50,60,70,80,80", audiogram_r="30,35,40,55,60,65,75,80",
                  instruments="Flute,Oboe", gains="1.5,-2", residual="0.25").status_code == 200
    row = wait_done(client)[0]
    assert row["residual"] == 0.25 and row["instruments"] == ["Flute", "Oboe"] and row["gains"] == [1.5, -2.0]
    assert row["audiogram_l"] == [25, 30, 40, 50, 60, 70, 80, 80]
    assert row["audiogram_r"] == [30, 35, 40, 55, 60, 65, 75, 80]
