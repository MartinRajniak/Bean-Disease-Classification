# Bean Disease Classification
A complete end-to-end ML system for bean disease classification using GPU-accelerated training, MLflow tracking, and Airflow orchestration.

## Table of Contents
- [Getting Started](#getting-started)
- [Features](#features)
- [Notebooks](#notebooks)
- [Training Options](#training-options)
- [Local Development Setup](#local-development-setup)
- [Project Structure](#project-structure)
- [Experiment Tracking & Orchestration](#experiment-tracking--orchestration)
- [Debugging](#debugging)
- [Technology Stack](#technology-stack)
- [Troubleshooting](#troubleshooting)

## Getting Started

**New to this project?** Follow this path:

1. **Learn**: Read [`bean_disease_classification.ipynb`](bean_disease_classification.ipynb) to understand the problem and ML approach
2. **Train**: Upload [`train_on_colab.ipynb`](train_on_colab.ipynb) to Google Colab for free GPU training (~15-30 min)
3. **Compare**: Use [`test_model_performance.ipynb`](test_model_performance.ipynb) to evaluate different models
4. **Production**: Set up [local Docker environment](#local-development-setup) for MLflow tracking and Airflow orchestration

**Just want to train quickly?** → Use [`train_on_colab.ipynb`](train_on_colab.ipynb) (no setup required)

**Building production pipelines?** → Set up [local environment](#local-development-setup) for REST API + MLflow + Airflow

## Features

### Core Capabilities
- **GPU-Accelerated Training**: CUDA-enabled TensorFlow training with automatic memory management
- **Experiment Tracking**: MLflow for logging parameters, metrics, and model artifacts
- **Pipeline Orchestration**: Airflow for automated workflow management
- **REST API**: Flask-based training service with configurable hyperparameters
- **Production Code**: Migrated from Jupyter notebook to modular, testable source code
- **Mobile Deployment**: Automatic TFLite conversion for Android integration

### Technical Highlights
- **Docker Image Optimization**: Multi-stage build (dependencies cached, code rebuilds in ~30s)
- **Transfer Learning**: Xception/EfficientNetV2/MobileNet with two-phase training
- **Data Pipeline**: Stratified splitting with TensorFlow Datasets
- **Persistent Storage**: All experiments and artifacts survive container restarts

## Notebooks

### 📚 `bean_disease_classification.ipynb` - **Start Here**

Complete tutorial explaining the problem, data, and training methodology. Covers:
- Why bean disease detection matters for African food security
- Data exploration and preprocessing
- Transfer learning with two-phase training (freeze → fine-tune)
- Model evaluation and mobile deployment

**Use this notebook to**: Understand the problem domain and ML approach before training models.

### 🔬 `test_model_performance.ipynb`

Compare trained models across architectures (Xception, MobileNet) and formats (Keras, TFLite). Includes accuracy, F1-score, inference time, model size, and degradation analysis.

**Example**: MobileNet TFLite is 91.5% smaller and 2.15x faster with only 6.49% accuracy loss vs Xception.

**Use this notebook to**: Choose the best model for your deployment constraints (accuracy vs size vs speed).

## Training Options

| Method | GPU | Cost | Setup | MLflow | Best For |
|--------|-----|------|-------|--------|----------|
| **Local (REST API)** | Your GPU | Free | Docker + NVIDIA | ✅ Yes | Production pipelines & automation |
| **Colab (`train_on_colab.ipynb`)** | Free T4 | Free | None | ❌ No | Quick experiments without local GPU |
| **Cloud (GCP)** | Cloud GPU | ~$1-2/hr | Medium | ✅ Yes | Production training at scale |

### Local Training via REST API

Train using the GPU-accelerated Flask API (requires [local setup](#local-development-setup)):

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "epochs_pretrain": 5,
    "epochs_finetune": 10,
    "initial_lr": 0.1,
    "finetune_lr": 0.01,
    "batch_size": 16,
    "experiment_name": "bean_disease_api",
    "run_name": "training_20250107_001"
  }'
```

Results tracked in MLflow UI at http://localhost:5000

### Google Colab Training

Upload `train_on_colab.ipynb` to Colab, enable GPU (Runtime → Change runtime type → GPU), and run all cells. Training takes ~15-30 minutes and produces downloadable Keras + TFLite models.

**Note**: Uses same `train_model_core()` code as local/cloud for consistency.

### Cloud Training (GCP)

Coming soon.

## Local Development Setup

### Prerequisites

- Docker and Docker Compose
- **NVIDIA Container Toolkit** - Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- NVIDIA GPU with CUDA support
- ~5GB free disk space
- Ports 5000, 8000, 8080 available

### Build & Start

```bash
# Build training image (recommended method)
./scripts/docker_build.sh

# Start all services
./scripts/start_local.sh

# Check status
docker-compose -f docker-compose.local.yml ps
```

**Build optimization**: Dependencies cached (~10 min), code rebuilds in ~30 sec.

### Services

Access the running services:
- **Training API**: http://localhost:8000/health
- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080 (admin/admin)

### Stop & Cleanup

```bash
# Stop services (data persists)
./scripts/stop_local.sh

# Stop and remove all data
docker-compose -f docker-compose.local.yml down -v
```

**Data storage**: `./local_data/mlflow/` (experiments), `./local_data/airflow_data/` (metadata), `./dags/` (pipelines)

## Project Structure

```
bean-disease-classification/
├── Dockerfile                         # Training service image (multi-stage)
├── requirements.txt                   # Python dependencies
├── docker-compose.local.yml          # Local environment configuration
├── bean_disease_classification.ipynb # Main tutorial notebook (start here)
├── train_on_colab.ipynb              # Google Colab training notebook
├── test_model_performance.ipynb      # Model comparison and evaluation
├── src/                               # Production code (from notebook)
│   ├── config/
│   │   └── training_config.py         # Hyperparameters & settings
│   ├── data/
│   │   └── data_loader.py             # Dataset loading & preprocessing
│   ├── models/
│   │   └── model_builder.py           # Transfer learning architecture
│   └── training/
│       ├── api.py                     # Flask REST API
│       └── trainer.py                 # Training logic & MLflow integration
├── dags/                              # Airflow DAG definitions
├── scripts/                           # Utility scripts
│   ├── docker_build.sh                # Build Docker image
│   ├── docker_run.sh                  # Run scripts in container
│   ├── start_local.sh                 # Start environment
│   └── stop_local.sh                  # Stop environment
└── local_data/                        # Persistent data (gitignored)
```

## Experiment Tracking & Orchestration

### MLflow Experiment Tracking

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("bean_disease_classification")

with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")
```

View experiments at http://localhost:5000

### Airflow Pipeline Orchestration

Create DAG files in `./dags/` to automate training workflows. Training service is pre-configured as an HTTP connection for triggering jobs from Airflow.

## Debugging

### View Logs
```bash
# Follow training service logs
docker-compose -f docker-compose.local.yml logs -f training-service

# View last 100 lines
docker-compose -f docker-compose.local.yml logs --tail=100 training-service
```

### Restart Service
```bash
# Restart training service (e.g., after code changes)
docker-compose -f docker-compose.local.yml restart training-service

# Full restart
docker-compose -f docker-compose.local.yml down && docker-compose -f docker-compose.local.yml up -d
```

### Interactive Debugging
```bash
# Run container with GPU and shell access
docker run --gpus all --rm -it \
  -v $(pwd)/src:/app/src \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  bean-disease-classification-image:latest \
  bash

# Run script in container with GPU
./scripts/docker_run.sh src/training/trainer.py
```

## Technology Stack

**Core Components:**
- TensorFlow 2.19.0 with CUDA - GPU-accelerated deep learning
- MLflow 3.4.0 - Experiment tracking and model registry
- Apache Airflow 2.7.1 - Workflow orchestration
- Flask + Gunicorn - Production API server
- Docker + NVIDIA Container Toolkit - GPU containerization

**ML Pipeline:**
- Transfer learning with ImageNet pretrained models
- Stratified data splitting for balanced training
- Two-phase training (freeze → fine-tune)
- TFLite conversion for mobile deployment

## Troubleshooting

### GPU Issues
- **GPU not detected**: Install NVIDIA Container Toolkit, verify with `docker run --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`
- **TensorFlow GPU errors**: Check CUDA compatibility and GPU memory (`nvidia-smi`)
- **"Dst tensor is not initialized"**: Fixed in latest code, rebuild image with `./scripts/docker_build.sh`

### Service Issues
- **Port conflicts**: Ensure ports 5000, 8000, 8080 are free
- **Service not starting**: Wait 2-3 minutes, check logs with `docker-compose -f docker-compose.local.yml logs [service]`
- **Training fails**: Check `docker-compose -f docker-compose.local.yml logs training-service`

### Build Issues
- **Slow builds**: Use `./scripts/docker_build.sh` - code changes rebuild in ~30s
- **Full rebuild needed**: Change in requirements.txt triggers ~10 min rebuild

### Useful Commands
```bash
# Container status
docker ps

# Service logs
docker-compose -f docker-compose.local.yml logs -f [service_name]

# Restart service
docker-compose -f docker-compose.local.yml restart [service_name]

# GPU check
nvidia-smi

# Rebuild and restart training service
./scripts/docker_build.sh
docker-compose -f docker-compose.local.yml up -d --force-recreate training-service
```
