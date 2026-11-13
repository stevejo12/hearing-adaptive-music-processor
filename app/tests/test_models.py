from cadenza_app import models


def test_downloads_only_missing_models():
    cached = {"cadenzachallenge/ConvTasNet_Flute_NonCausal", "cadenzachallenge/ConvTasNet_Oboe_NonCausal"}
    calls = []

    def fake_fetch(repo, local_files_only=False):
        calls.append((repo, local_files_only))
        if local_files_only and repo not in cached:
            raise FileNotFoundError(repo)
        return "/x/" + repo

    models.ensure_models(fetch=fake_fetch)
    downloaded = [r for r, local in calls if not local]
    assert set(downloaded) == set(models.repos().values()) - cached
    assert models.status()["state"] == "ready" and models.is_ready()


def test_nothing_downloaded_when_all_cached():
    calls = []

    def fake_fetch(repo, local_files_only=False):
        calls.append(local_files_only)
        return "/x"

    models.ensure_models(fetch=fake_fetch)
    assert all(calls) and models.is_ready()   # only local checks, no download


def test_download_failure_is_reported():
    def fake_fetch(repo, local_files_only=False):
        raise OSError("offline")

    models.ensure_models(fetch=fake_fetch)
    st = models.status()
    assert st["state"] == "error" and "offline" in st["error"] and not models.is_ready()
