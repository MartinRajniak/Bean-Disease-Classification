# Bean Disease Classification
A complete end-to-end ML system for bean disease classification using GPU-accelerated training, MLflow tracking, and Airflow orchestration.

## Table of Contents
- [Features](#features)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
  - [Build the Training Image](#build-the-training-image)
  - [Start the Environment](#start-the-environment)
  - [Access Services](#access-services)
- [Project Structure](#project-structure)
- [Usage](#usage)
  - [Training Models](#training-models)
  - [Experiment Tracking](#experiment-tracking)
  - [Pipeline Orchestration](#pipeline-orchestration)
- [Debugging](#debugging)
- [Technology Stack](#technology-stack)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

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

## System Architecture

**Services:**
- **MLflow** (http://localhost:5000) - Experiment tracking and model registry
- **Airflow** (http://localhost:8080) - Pipeline orchestration (admin/admin)
- **Training Service** (http://localhost:8000) - GPU-accelerated training API

**Storage:**
- `./local_data/mlflow/` - Experiments, models, artifacts
- `./local_data/airflow_data/` - Airflow metadata
- `./dags/` - Pipeline definitions (version controlled)

## Prerequisites

- Docker and Docker Compose
- **NVIDIA Container Toolkit** (required for GPU support)
  - Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
- NVIDIA GPU with CUDA support
- ~5GB free disk space
- Ports 5000, 8000, 8080 available

## Quick Start

### Build the Training Image

```bash
# Using build script (recommended)
./scripts/docker_build.sh

# Or manually
docker build -t bean-disease-classification-image:latest .
```

**Build Optimization**: The Dockerfile uses a two-stage build:
1. **Base layer**: Python dependencies (~10 min, cached until requirements.txt changes)
2. **Code layer**: Source code (~30 sec, rebuilt on code changes)

### Start the Environment

```bash
# Start all services (recommended - includes health checks)
./scripts/start_local.sh

# Or use Docker Compose directly
docker-compose -f docker-compose.local.yml up -d

# Check status
docker-compose -f docker-compose.local.yml ps

# Stop services (data persists)
./scripts/stop_local.sh
# or
docker-compose -f docker-compose.local.yml down

# Stop and remove all data
docker-compose -f docker-compose.local.yml down -v
```

**Note**: The `--gpus all` flag is configured in docker-compose.yml for GPU access.

### Access Services

- **Training API**: http://localhost:8000/health
- **MLflow UI**: http://localhost:5000
- **Airflow UI**: http://localhost:8080 (username: `admin`, password: `admin`)

## Project Structure

```
bean-disease-classification/
├── Dockerfile                         # Training service image (multi-stage)
├── requirements.txt                   # Python dependencies
├── docker-compose.local.yml          # Local environment configuration
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

## Usage

### Training Models

Train models via REST API:

```bash
# Start training with custom parameters
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

# Check service health
curl http://localhost:8000/health
```

**Training Pipeline:**
- Data: TensorFlow Datasets (beans) with stratified split
- Model: Transfer learning (Xception/EfficientNetV2/MobileNet)
- Training: Phase 1 (frozen base) → Phase 2 (fine-tuning)
- Output: Keras model + TFLite for mobile

### Experiment Tracking

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

### Pipeline Orchestration

- Create DAG files in `./dags/`
- Trigger and monitor runs in Airflow UI
- Training service pre-configured as HTTP connection

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

## Roadmap

### Phase 1: ML Training Pipeline ✅
- ✅ Local GPU-accelerated environment
- ✅ MLflow experiment tracking
- ✅ Airflow orchestration
- ✅ Production-ready code structure
- ✅ TFLite mobile deployment

### Phase 2: Automation & Optimization (Current)
- [ ] Automated training DAG
- [ ] Hyperparameter optimization
- [ ] Model comparison dashboard
- [ ] CI/CD pipeline

### Phase 3: Production Deployment
- [ ] Cloud training (GCP Cloud Run)
- [ ] Model serving API
- [ ] Android app integration
- [ ] A/B testing & monitoring

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

---

**Status**: Production Training Pipeline Ready
**Next**: Implement automated training DAG in Airflow
