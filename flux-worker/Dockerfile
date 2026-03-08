FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

COPY . /app

# Make start_runpod.sh executable
RUN chmod +x /app/start_runpod.sh

# NOTE:
# We intentionally do NOT download the gated Hugging Face model at build time,
# because the Docker build environment does not have your HF token and will
# fail with 401 Unauthorized. The model is downloaded at container startup via
# start_runpod.sh, which checks if the model exists in the RunPod volume and
# downloads it if needed. The model is then loaded in handler.py via load_model().

# Use start_runpod.sh as entrypoint to handle model download before starting the handler
ENTRYPOINT ["/app/start_runpod.sh"]

