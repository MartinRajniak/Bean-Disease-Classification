# Bean Disease Classification
A complete end-to-end machine learning system for bean disease classification using industry-standard tools and practices.

## System Architecture
Local Development Environment
- **MLflow Server (Experiment Tracking)**
  - **UI:** http://localhost:5000
  - **Backend:** SQLite database
  - **Artifacts:** Local file storage


- **Airflow Server (Pipeline Orchestration)**
  - **UI:** http://localhost:8080 (admin/admin)
  - **Backend:** SQLite database
  - **Mode:** Standalone (webserver + scheduler)

- **Persistent Data Storage**
  - ./local_data/mlflow/ (experiments, models, artifacts)
  - DAGs in ./dags/ (version controlled)

## Current Status

### What's Working ✅
- **MLflow Server**: Fully operational for experiment tracking and model registry
- **Airflow Server**: Running with web UI and scheduler for pipeline orchestration
- **Data Persistence**: All experiments and metadata persist between restarts
- **Docker Integration**: Containerized services with proper networking
- **Development Ready**: Environment ready for ML pipeline development

### What's Implemented
- Local development environment with Docker Compose
- MLflow server with SQLite backend for lightweight development
- Airflow server with SQLite backend in standalone mode
- Proper container networking (Airflow can communicate with MLflow)
- Persistent data volumes for stateful development

## Quick Start

### Prerequisites
- Docker and Docker Compose installed
- ~2GB free disk space
- Ports 5000 and 8080 available

### Start the Environment
```bash
# Start all services
docker-compose -f docker-compose.local.yml up -d

# Check status
docker-compose -f docker-compose.local.yml ps

# View logs
docker-compose -f docker-compose.local.yml logs -f
```

### Access the UIs
- MLflow UI: http://localhost:5000
- Airflow UI: http://localhost:8080 (username: admin, password: admin)

### Stop the Environment
```bash
# Stop services (data persists)
docker-compose -f docker-compose.local.yml down

# Stop and remove all data (clean slate)
docker-compose -f docker-compose.local.yml down -v
```

## Project Structure
```
bean-disease-ml-pipeline/
├── README.md                          # This file
├── docker-compose.local.yml          # Local development environment
├── local_data/                        # Persistent data (gitignored)
│   └── mlflow/                        # MLflow experiments and artifacts
├── dags/                              # Airflow DAGs (version controlled)
│   └── test_pipeline.py               # Example pipeline
├── src/                               # ML source code
│   ├── training/                      # Training modules
│   ├── data/                          # Data processing
│   └── utils/                         # Utilities
├── configs/                           # Experiment configurations
└── scripts/                           # Utility scripts
    ├── start_local.sh                 # Environment startup
    └── stop_local.sh                  # Environment shutdown
```

## Development Workflow

### 1. Experiment Tracking (MLflow)
```python
import mlflow

# Set tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

# Create experiment
mlflow.set_experiment("bean_disease_classification")

# Log experiment
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.001)
    mlflow.log_metric("accuracy", 0.95)
    mlflow.log_artifact("model.pkl")
```

### 2. Pipeline Development (Airflow)
- Create DAG files in `./dags/` directory
- Access Airflow UI to manage and monitor pipelines
- Trigger manual runs for testing
- View logs and task status in the web interface

## Technology Stack

### Core Components
- **MLflow 3.1.4:** Experiment tracking, model registry, and deployment
- **Apache Airflow 2.7.1:** Workflow orchestration and scheduling
- **SQLite:** Lightweight database for development
- **Docker & Docker Compose:** Containerization and service orchestration

### Development Environment
- **Local-first:** No cloud dependencies for development
- **Persistent:** Data survives container restarts
- **Lightweight:** Minimal resource requirements
- **Production-ready:** Uses same tools as production ML systems

## Next Steps (Roadmap)

### Phase 1: Basic ML Pipeline (Current)
- ✅ Local development environment
- ✅ MLflow experiment tracking
- ✅ Airflow pipeline orchestration
- Basic bean disease classification DAG
- Synthetic data training pipeline

### Phase 2: Real ML Implementation
- Bean disease dataset integration
- CNN model training pipeline
- Hyperparameter optimization
- Model evaluation and validation
- TFLite conversion for mobile

### Phase 3: Production Features
- Cloud training integration (GCP)
- Model serving API
- Android app integration
- CI/CD pipeline
- Monitoring and alerting

## Cost & Resource Usage

### Local Development
- **Cost:** $0 (runs entirely on your machine)
- **CPU:** ~1-2 cores when active
- **Memory:** ~2-4GB RAM
- **Storage:** ~1-5GB for experiments and models

### Advantages
- No cloud costs during development
- Full control over environment
- Works offline
- Industry-standard tools and practices
- Easy to scale to cloud when ready

## Troubleshooting

### Common Issues
- **Port conflicts:** Ensure ports 5000 and 8080 are free
- **Docker issues:** Restart Docker Desktop if containers won't start
- **Permission errors:** Check that Docker has file system access
- **Service not accessible:** Wait 2-3 minutes for services to fully start

### Useful Commands
```bash
# Check container status
docker ps

# View service logs
docker-compose -f docker-compose.local.yml logs [service_name]

# Restart specific service
docker-compose -f docker-compose.local.yml restart [service_name]

# Clean restart
docker-compose -f docker-compose.local.yml down && docker-compose -f docker-compose.local.yml up -d
```

## Learning Objectives
This setup teaches industry-standard ML engineering practices:
- **Experiment Management:** Track and compare ML experiments systematically
- **Pipeline Orchestration:** Automate ML workflows with proper dependency management
- **Model Registry:** Manage model versions and deployments
- **Infrastructure as Code:** Reproducible development environments
- **Production Readiness:** Tools and patterns used in real ML systems

---

**Status:** Development Environment Ready  
**Next:** Implement first ML training pipeline  