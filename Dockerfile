# Last NVIDIA image that supports my GPU (Quadro P600) - however is bundled with only Tensorfow 2.12
# FROM nvcr.io/nvidia/tensorflow:23.05-tf2-py3

# Last official image that works (https://github.com/tensorflow/tensorflow/issues/80538) (fastest)
# FROM tensorflow/tensorflow:2.17.0-gpu

# 2.20.0 doesn't work (compatibility issues) even with hotfix
# FROM tensorflow/tensorflow:2.20.0-gpu

# Works only with the hotfix which installs tensorflow[and-cuda] as part of requirements.txt (inefficient)
# FROM tensorflow/tensorflow:2.19.0-gpu

FROM python:3.12-slim AS base

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends build-essential

# Clean up apt caches in the final layer (outside the mount)
RUN apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r /tmp/requirements.txt

FROM base AS development
WORKDIR /app
COPY src/ /app/src/
CMD ["python", "-c", "print('ML Training container ready')"]