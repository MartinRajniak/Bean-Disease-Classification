# Bean Disease Classification

End-to-end ML pipeline for bean disease classification with GPU-accelerated training, experiment tracking, workflow orchestration, and mobile deployment.

## Table of Contents
- [Getting Started](#getting-started)
- [Training Options](#training-options)
- [Notebooks](#notebooks)
- [Mobile App](#mobile-app)
- [Local Development Setup](#local-development-setup)
- [Project Structure](#project-structure)
- [Experiment Tracking & Orchestration](#experiment-tracking--orchestration)
- [Debugging](#debugging)
- [Technology Stack](#technology-stack)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

## Getting Started

**Choose your training approach:**

| Your Situation | Recommended Method | Details |
|----------------|-------------------|---------|
| No GPU, want automation | [Modal](#modal-cloud-training) | $30/month free credits, CLI-based |
| No GPU, manual experiments | [Colab](#google-colab-training) | Free, browser-based |
| Have local GPU | [Local Docker](#local-development-setup) | Free, full MLflow/Airflow stack |

**Learning path:**
1. Read [`bean_disease_classification.ipynb`](bean_disease_classification.ipynb) - Understand the ML approach
2. Choose a training method above - Train your first model
3. Use [`test_model_performance.ipynb`](test_model_performance.ipynb) - Compare model performance
4. Try the [Mobile App](#mobile-app) - Test on real images

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

Complete tutorial explaining the problem, data, and training methodology:
- Why bean disease detection matters for African food security
- Data exploration and preprocessing
- Transfer learning with two-phase training (freeze → fine-tune)
- Model evaluation and mobile deployment

**Use for**: Understanding the ML approach and problem domain.

### 🚀 `train_on_colab.ipynb`

Google Colab notebook for training models on free cloud GPUs:
- No local setup required - runs entirely in browser
- Same training code as production (`train_model_core()`)
- Configurable hyperparameters (model, epochs, learning rates)
- Downloads trained Keras + TFLite models

**Use for**: Quick training without local GPU (Runtime → GPU → Run all).

### 🔬 `test_model_performance.ipynb`

Benchmark and compare trained models:
- Accuracy, F1-score across architectures (Xception, MobileNet, etc.)
- Inference time and model size analysis
- TFLite conversion degradation metrics

**Example**: MobileNet TFLite is 91.5% smaller and 2.15x faster with only 6.49% accuracy loss vs Xception.

**Use for**: Choosing optimal model for deployment constraints.

## Training Options

| Method | GPU | Cost | Setup | MLflow | Automatable | Best For |
|--------|-----|------|-------|--------|-------------|----------|
| **Local (REST API)** | Your GPU | Free | Docker + NVIDIA | ✅ Yes | ✅ Yes | Production pipelines with local GPU |
| **Modal (`modal_train.py`)** | T4/A10G/A100 | $30/month free* | Minimal | ❌ No | ✅ Yes | Automated training without local GPU |
| **Colab (`train_on_colab.ipynb`)** | Free T4 | Free | None | ❌ No | ❌ No | Quick manual experiments |
| **Cloud (GCP)** | Cloud GPU | ~$1-2/hr | Medium | ✅ Yes | ✅ Yes | Production training at scale |

\* Modal Starter plan includes $30 of free compute credits per month (~15-30 hours of T4 GPU time)

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

### Modal Cloud Training

Train on Modal's cloud GPUs with pay-as-you-go pricing. **Includes $30/month free credits on Starter plan** - perfect for automated training without local GPU.

**Prerequisites:**
```bash
# Install Modal
pip install modal

# Authenticate (one-time)
modal token new
```

**Run Training:**
```bash
# Default: MobileNet on T4 GPU
modal run modal_train.py

# Custom model and GPU
modal run modal_train.py --base-model XCEPTION --gpu A100-40GB

# Custom epochs
modal run modal_train.py --epochs-pretrain 8 --epochs-finetune 15
```

**Available Options:**
- **Models**: `MOBILE_NET`, `XCEPTION`, `EFFICIENT_NET_V2`
- **GPUs**: `T4` (~$0.60/hr), `A10G` (~$1.10/hr), `A100-40GB` (~$2.50/hr), `A100-80GB`

**Important**: Modal uses the Docker image from GitHub Container Registry (`ghcr.io/martinrajniak/bean-disease-classification-image:latest`). To train with your latest code changes:
1. Push changes to GitHub
2. Run the `build_and_push_docker` GitHub Action to build new image
3. Wait for build to complete (~10 min)
4. Run `modal run modal_train.py`

Models download automatically to `models/modal_{model}_{timestamp}/` after training.

**Why Modal?**
- ✅ Free tier ($30/month credits) vs paid-only cloud providers
- ✅ Automatable via CLI/Python (unlike Colab)
- ✅ No local GPU required
- ✅ Dataset caching (faster subsequent runs)
- ✅ Access to faster GPUs (A100) vs Colab's T4

### Google Colab Training

Upload `train_on_colab.ipynb` to Colab, enable GPU (Runtime → Change runtime type → GPU), and run all cells. Training takes ~15-30 minutes and produces downloadable Keras + TFLite models.

**Note**: Uses same `train_model_core()` code as local/cloud for consistency.

### Cloud Training (GCP)

Coming soon.

## Mobile App

Android application for real-time bean disease detection using trained TFLite models.

**Location**: [`mobile/`](mobile/) directory (Kotlin/Android)

**Features**:
- Camera integration for live disease detection
- Offline inference using TFLite model
- Automatic model download from GitHub releases
- Displays disease predictions with confidence scores

**Get Started**: Download the latest APK from [GitHub Releases](https://github.com/martinr92/Bean-Disease-Classification/releases) or build from source in the `mobile/` directory.

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
├── modal_train.py                    # Modal cloud training script
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

**ML/Training:**
- TensorFlow 2.19.0 with CUDA - GPU-accelerated deep learning
- Transfer learning - Xception/EfficientNetV2/MobileNet (ImageNet pretrained)
- Two-phase training - Freeze base → Fine-tune layers
- TFLite conversion - Mobile deployment optimization

**Infrastructure:**
- Docker + NVIDIA Container Toolkit - GPU containerization
- Modal - Serverless cloud GPU compute
- MLflow 3.4.0 - Experiment tracking and model registry
- Apache Airflow 2.7.1 - Workflow orchestration
- Flask + Gunicorn - Production REST API

**Mobile:**
- Kotlin/Android - Native mobile app
- TFLite - On-device inference

## Roadmap

### 🎯 Milestone 1: Automated MLOps Pipeline
**Goal**: End-to-end automation from training → evaluation → deployment

**Tasks**:
- [ ] GitHub Actions workflow for scheduled/triggered training on Modal
- [ ] Automated model evaluation against baseline metrics
- [ ] Auto-promotion to production if performance improves
- [ ] Auto-create GitHub releases with TFLite models
- [ ] Mobile app auto-update mechanism for latest models
- [ ] Notification system for training completion/failures

**Impact**: Continuous model improvement without manual intervention

---

### 🎯 Milestone 2: Production Model Monitoring & Drift Detection
**Goal**: Real-time visibility into production model performance

**Tasks**:
- [ ] Prediction logging API endpoint (mobile app → backend)
- [ ] Metrics dashboard: prediction distribution, confidence scores, error rates
- [ ] Data drift detection (compare production vs training distribution)
- [ ] Model performance monitoring (accuracy degradation alerts)
- [ ] Auto-trigger retraining on drift/performance drop
- [ ] A/B testing framework for comparing model versions

**Tools**: Evidently AI, WhyLabs, or Grafana dashboard

**Impact**: Early detection of model degradation, data-driven retraining decisions

---

### 🎯 Milestone 3: Interactive Web Demo
**Goal**: Browser-based demo for easy access and showcasing

**Tasks**:
- [ ] Gradio interface for image upload → prediction
- [ ] Deploy to HuggingFace Spaces (free hosting)
- [ ] Display confidence scores and prediction explanations
- [ ] Example image gallery with pre-loaded samples
- [ ] User feedback collection mechanism
- [ ] Integration with README and project landing page

**Impact**: Increased visibility, user feedback collection, portfolio showcase

---

### 🎯 Milestone 4: Comprehensive Testing Suite
**Goal**: Ensure code quality and prevent regressions

**Tasks**:
- [ ] Unit tests: data loading, preprocessing, model architecture
- [ ] Integration tests: training pipeline, API endpoints, MLflow tracking
- [ ] Model regression tests: performance benchmarks on test set
- [ ] GitHub Actions CI: run tests on every PR
- [ ] Test coverage reporting (target: >80%)
- [ ] Pre-commit hooks for linting and formatting

**Tools**: pytest, coverage.py, pre-commit

**Impact**: Reliable codebase, safe refactoring, faster development

---

### 🎯 Milestone 5: Multi-Crop Disease Detection
**Goal**: Expand to multiple crops for broader agricultural impact

**Tasks**:
- [ ] Integrate PlantVillage dataset (14 crops, 38 diseases)
- [ ] Implement multi-task model or crop-specific models
- [ ] Hierarchical classification: crop type → disease detection
- [ ] Update mobile app UI for crop selection
- [ ] Comparative benchmarks across crops
- [ ] Transfer learning experiments: bean model → other crops

**Impact**: 10x more valuable for farmers, more complex ML problem

---

### 💡 Future Enhancements

**Data & Training**:
- Data versioning with DVC
- Automated hyperparameter tuning (Optuna)
- Active learning pipeline for hard examples
- Synthetic data generation for rare diseases

**Model Explainability**:
- Grad-CAM visualizations (which leaf parts influenced prediction)
- SHAP values for feature importance
- Confidence calibration analysis

**Deployment**:
- iOS mobile app
- Raspberry Pi edge deployment kit for offline field use
- WhatsApp bot integration for farmers
- Progressive Web App (PWA) version

**API & Documentation**:
- OpenAPI/Swagger documentation
- Public API with rate limiting
- SDK/client libraries (Python, JavaScript)

**Infrastructure**:
- Kubernetes deployment for production API
- Auto-scaling based on traffic
- Multi-region deployment for low latency

---

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
