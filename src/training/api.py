"""
Training API that mimics Cloud Run deployment
This same code can run locally or in Cloud Run
"""

from flask import Flask, request, jsonify
import os
import json
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from datetime import datetime
import traceback

app = Flask(__name__)


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "training-api"})


@app.route("/train", methods=["POST"])
def train_model():
    """Train model endpoint - same interface as Cloud Run"""
    try:
        # Get training configuration from request
        config = request.get_json() or {}

        # Set defaults
        training_config = {
            "epochs": config.get("epochs", 5),
            "learning_rate": config.get("learning_rate", 0.001),
            "batch_size": config.get("batch_size", 32),
            "experiment_name": config.get("experiment_name", "api_training"),
            "run_name": config.get(
                "run_name", f'run_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
            ),
        }

        print(f"Starting training with config: {training_config}")

        # Connect to MLflow
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment(training_config["experiment_name"])

        with mlflow.start_run(run_name=training_config["run_name"]):
            # Log parameters
            mlflow.log_params(training_config)
            mlflow.log_param("training_service", "local_api")

            # Generate synthetic data (replace with your data loading)
            x_train, y_train, x_val, y_val = load_data()

            # Build and compile model
            model = build_model()
            model.compile(
                optimizer=tf.keras.optimizers.Adam(
                    learning_rate=training_config["learning_rate"]
                ),
                loss="categorical_crossentropy",
                metrics=["accuracy"],
            )

            # Train model
            history = model.fit(
                x_train,
                y_train,
                epochs=training_config["epochs"],
                batch_size=training_config["batch_size"],
                validation_data=(x_val, y_val),
                verbose=1,
            )

            # Log metrics
            final_metrics = {
                "accuracy": float(history.history["accuracy"][-1]),
                "val_accuracy": float(history.history["val_accuracy"][-1]),
                "loss": float(history.history["loss"][-1]),
                "val_loss": float(history.history["val_loss"][-1]),
            }

            mlflow.log_metrics(final_metrics)

            # Save model
            mlflow.tensorflow.log_model(model, "model")

            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            # Log TFLite model
            tflite_path = "/tmp/model.tflite"
            with open(tflite_path, "wb") as f:
                f.write(tflite_model)
            mlflow.log_artifact(tflite_path, "tflite_model")

            model_size_mb = len(tflite_model) / (1024 * 1024)
            mlflow.log_metric("tflite_size_mb", model_size_mb)

            result = {
                "status": "success",
                "mlflow_run_id": mlflow.active_run().info.run_id,
                "metrics": final_metrics,
                "tflite_size_mb": model_size_mb,
                "message": "Training completed successfully",
            }

            print(f"Training completed: {result}")
            return jsonify(result)

    except Exception as e:
        error_result = {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"Training failed: {error_result}")
        return jsonify(error_result), 500


def load_data():
    """Load training data"""
    # Synthetic data for now
    x_train = np.random.rand(800, 224, 224, 3)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 3, 800), 3)
    x_val = np.random.rand(200, 224, 224, 3)
    y_val = tf.keras.utils.to_categorical(np.random.randint(0, 3, 200), 3)
    return x_train, y_train, x_val, y_val


def build_model():
    """Build CNN model"""
    return tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(224, 224, 3)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(3, activation="softmax"),
        ]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
