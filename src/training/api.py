from flask import Flask, request, jsonify
import traceback
from datetime import datetime

import sys
import os

# Add 'src' to the path so that we can use absolute path in imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config.training_config import TrainingConfig
from training.trainer import train_model

app = Flask(__name__)


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

        result = train_model(config)
        return jsonify(result)
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
