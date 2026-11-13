# syntax=docker/dockerfile:1
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

# --- Cadenza / Clarity baseline, pinned. Change CLARITY_REF -> image must be rebuilt.
ARG CLARITY_REF=9df6486fb0bddc7619b3b99f1b3a5c72c109a3ec
RUN git clone https://github.com/claritychallenge/clarity.git /opt/clarity \
    && git -C /opt/clarity checkout ${CLARITY_REF}

# CPU-only PyTorch (a few hundred MB instead of several GB for the CUDA build)
RUN pip install --no-cache-dir torch torchaudio --index-url https://download.pytorch.org/whl/cpu
# pyflac has no arm64 wheel and compiles from source, so a compiler is needed while installing
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libffi-dev libogg-dev \
    && pip install --no-cache-dir /opt/clarity \
    && apt-get purge -y build-essential libffi-dev && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# `recipes.cad2...` (ConvTasNet + baseline code) lives in the Clarity repo, not the pip package
# The pretrained Cadenza models are NOT in the image: on start-up the app checks /data/hf and
# downloads only what is missing (first start only), so rebuilds never re-download them.
ENV PYTHONPATH=/opt/clarity \
    HF_HOME=/data/hf \
    PYTHONUNBUFFERED=1
ARG CADENZA_CAUSALITY=noncausal
ENV CADENZA_CAUSALITY=${CADENZA_CAUSALITY}

WORKDIR /srv
COPY cadenza_app ./cadenza_app

ENV DATA_DIR=/data
VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/healthz')"
CMD ["uvicorn", "cadenza_app.main:app", "--host", "0.0.0.0", "--port", "8000"]
