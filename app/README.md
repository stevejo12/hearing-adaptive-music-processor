# Cadenza web app (Docker)

A local web app around the Cadenza CAD2 Task 2 baseline. Log in, drop a woodwind-trio song and a
hearing profile (audiogram for each ear), press **Go**, and the history table below lists every
song with a **Before** player, an **After** player, a 5-star **Rate** and a **Comment** box.

## Run

```bash
cd app
make up                 # first build downloads Clarity, PyTorch (CPU) and the pretrained models
# open http://localhost:8080   (login: admin / admin)
```

Real login for anything beyond your own machine (env vars, picked up by compose):

```bash
APP_USER=me APP_PASSWORD='something long' APP_SECRET=$(openssl rand -hex 32) make up
```

Other settings (all optional): `PORT` (default 8080), `MAX_UPLOAD_MB` (50, songs), `MAX_MODEL_MB` (2048, model zips), `MAX_CACHED_MODELS` (2, models kept in memory), `MAX_SECONDS` (120,
songs are cut to this length to keep CPU time sane), `CADENZA_CAUSALITY` (`noncausal`/`causal`; changes which models are downloaded).

History (database + audio), the downloaded models (`data/hf`) and the login signing key are kept in `app/data/` on the host, so they survive restarts and rebuilds.

## Models: downloaded once, kept on the server

The pretrained Cadenza ConvTasNet models (`cadenzachallenge/ConvTasNet_<instrument>_NonCausal` for
Bassoon, Clarinet, Flute, Oboe, Sax, about 130 MB) are **not** inside the image. On every start the app
checks `data/hf/` and downloads only the models that are missing. While that runs, the page shows a
banner ("Downloading the pretrained Cadenza models (2 of 5: ...)") and **Go** stays disabled. After the
first start nothing is downloaded again, even after `make rebuild`, because `data/` is a persistent
volume. On the Proxmox server keep `app/data/` on storage that survives container re-creation.

The Clarity baseline code is baked into the image at one pinned commit (`CLARITY_REF` in the Dockerfile),
the same code the Colab notebooks run. The 13 GB Zenodo training/validation zips are not needed by the app.

## Adding a new model from the page (no rebuild)

At the top of the page, under **Model**, pick a model from the dropdown (it is remembered per browser)
and set the options (**Residual**: how much of the other instruments each separated stem keeps; the
baseline uses 0.1). To add a model, click **Upload model (.zip)**, optionally give it a name, and pick the
file. A progress bar shows the upload (models can be large; the limit is `MAX_MODEL_MB`, default 2048).
The server checks the zip, loads every network once to prove it works, and then stores it in
`data/models/`, so it stays available to everyone and survives restarts and rebuilds. **Delete model**
removes it for everyone (the built-in baseline cannot be deleted).

**Each model has its own table of songs.** The History table under the page shows only the songs made
with the model selected in the dropdown (before, after, rating and comment), so ratings of different models
never mix. Deleting a model also deletes its table (the page tells you how many songs go with it).

Zip layout (one folder per instrument, any subset of Bassoon, Clarinet, Flute, Oboe, Sax):

```
my_model.zip
  Flute/config.json
  Flute/model.safetensors
  Oboe/config.json
  Oboe/model.safetensors
```

Instruments missing from the zip use the baseline model. Only **safetensors** weights are accepted:
pickle-based files (`.pt`, `.bin`, `.ckpt`) can run code when loaded and are rejected. The folder format is
what `ConvTasNetStereo.save_pretrained(folder)` writes in Colab (`config.json` + `model.safetensors`).
If the server sits behind a reverse proxy, raise its upload size limit to match `MAX_MODEL_MB`.

## When the Cadenza part changes, rebuild

| Change | Command |
|---|---|
| App code (`cadenza_app/`) | `make rebuild` (fast) |
| New Clarity commit (`CLARITY_REF`) | `make rebuild` |
| Different or new models (edit `profile.INSTRUMENT_CENTRE_HZ` / `models.py`) | `make rebuild`; missing models download on the next start |
| Force a fresh Clarity download | `make rebuild-clean` |
| Force re-downloading the models | delete `data/hf/`, then restart |

A fine-tuned model from Colab needs no rebuild: zip it and upload it from the page.

## Layout

```
app/
  Dockerfile  docker-compose.yml  Makefile
  cadenza_app/
    main.py       # FastAPI: login, upload, history API, /api/status
    models.py     # start-up check + download of the pretrained models
    pipeline.py   # decode -> separate -> gains -> remix -> hearing aid (Clarity baseline functions)
    profile.py    # audiogram parsing, hearing-aid settings, default gains rule
    db.py         # SQLite history
    static/       # black-and-white page (index, login, css, js)
  tests/          # pytest: `make test`
```

## Known limits and assumptions

* **Hearing-aid settings** for a new audiogram follow the example in Clarity's `MultibandCompressor`
  docstring (ratio 4, make-up gain `max((HL-20)/3, 0)` per band). The CAD2 dataset ships ready-made
  values per listener that cannot be derived from a new audiogram, so check this against
  `compressor_params.*.json` in Colab before trusting the loudness of the output.
* **Default gains** come from a simple rule in `profile.py` (boost instruments that sit where the
  listener hears worst, limited to +/-6 dB). Gains can be typed in manually.
* Separation of real recordings is weak even for the baseline (the models were trained on synthetic data).
* One song is processed at a time. Expect minutes for a long song on CPU.
