# Last NVIDIA image that supports my GPU (Quadro P600) - however is bundled with only Tensorfow 2.12
# FROM nvcr.io/nvidia/tensorflow:23.05-tf2-py3

# Last official image that works (https://github.com/tensorflow/tensorflow/issues/80538)
# FROM tensorflow/tensorflow:2.17.0-gpu

# 2.20.0 doesn't work (compatibility issues) even with hotfix
# FROM tensorflow/tensorflow:2.20.0-gpu

# Works only with the hotfix which installs tensorflow[and-cuda] as part of requirements.txt
FROM tensorflow/tensorflow:2.19.0-gpu

# Install system dependencies (build-essential for some pip packages)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    # Add any other system libraries if needed by your specific Python packages
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements.txt file into the container
# It's good practice to do this before copying the rest of your code
# to leverage Docker's build cache.
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir: prevents pip from storing cache files, reducing image size
# -U: upgrade all specified packages to the newest available version
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir --ignore-installed -r requirements.txt

# Copy training code
COPY src/ /app/src/
WORKDIR /app

# Default command
CMD ["python", "-c", "print('ML Training container ready')"]