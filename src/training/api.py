from flask import Flask, request, jsonify
import traceback
from datetime import datetime
import mlflow
import mlflow.tensorflow

import sys
import os

# Add 'src' to the path so that we can use absolute path in imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.training_config import TrainingConfig
from training.trainer import train_model_core

app = Flask(__name__)


def train_model_with_mlflow(config: TrainingConfig):
    """
    Wrapper that adds MLflow tracking to core training function.
    Used by the API service for experiment tracking.
    """
    # Connect to MLflow
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.experiment_name)

    with mlflow.start_run(run_name=config.run_name):
        # Log configuration
        mlflow.log_params(config.to_dict())
        mlflow.log_param("training_service", "production_api")

        # Run core training (without MLflow)
        result = train_model_core(config)

        # Log metrics to MLflow
        mlflow.log_metrics(result["metrics"])

        # Save model to MLflow
        print("Saving model to MLflow...")
        mlflow.tensorflow.log_model(result["model"], "model")

        # Save TFLite model to MLflow
        tflite_path = "/tmp/bean_disease_model.tflite"
        with open(tflite_path, "wb") as f:
            f.write(result["tflite_model"])
        mlflow.log_artifact(tflite_path, "tflite_model")
        mlflow.log_metric("tflite_size_mb", result["tflite_size_mb"])

        # Add MLflow run ID to result
        result["mlflow_run_id"] = mlflow.active_run().info.run_id

        print(f"Training completed successfully with MLflow run: {result['mlflow_run_id']}")
        return result


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "bean-disease-training-api"})


@app.route("/train", methods=["POST"])
def train():
    try:
        # Get training configuration from request
        request_config = request.get_json() or {}

        # Create config object with overrides
        config = TrainingConfig(
            epochs_pretrain=int(request_config.get("epochs_pretrain", 5)),
            epochs_finetune=int(request_config.get("epochs_finetune", 10)),
            initial_lr=float(request_config.get("initial_lr", 0.1)),
            finetune_lr=float(request_config.get("finetune_lr", 0.01)),
            batch_size=int(request_config.get("batch_size", 16)),
            experiment_name=str(
                request_config.get("experiment_name", "bean_disease_api")
            ),
            run_name=str(
                request_config.get(
                    "run_name", f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
                )
            ),
            random_seed=int(request_config.get("random_seed", 42)),
        )

        print(f"Starting training with config: {config.to_dict()}")

        result = train_model_with_mlflow(config)

        # Remove model objects from response (not JSON serializable)
        response = {k: v for k, v in result.items() if k not in ["model", "tflite_model"]}

        return jsonify(response)
    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"Training failed: {error_result}")
        return jsonify(error_result), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
