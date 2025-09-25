import os
import json
import mlflow
import mlflow.tensorflow
import tensorflow as tf
import numpy as np
from datetime import datetime

def main():
    # Get config from environment variables (passed from Airflow)
    config = {
        'epochs': int(os.getenv('EPOCHS', '5')),
        'learning_rate': float(os.getenv('LEARNING_RATE', '0.001')),
        'batch_size': int(os.getenv('BATCH_SIZE', '32')),
        'experiment_name': os.getenv('EXPERIMENT_NAME', 'bean_disease_training'),
        'run_name': os.getenv('RUN_NAME', f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    }
    
    print(f"Starting training with config: {json.dumps(config, indent=2)}")
    
    # Connect to MLflow (host.docker.internal works from container to host)
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://host.docker.internal:5000')
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(config['experiment_name'])
    
    with mlflow.start_run(run_name=config['run_name']):
        # Log all parameters
        mlflow.log_params(config)
        mlflow.log_param("training_mode", "docker_container")
        
        # Generate synthetic bean disease data (replace with your real data loading)
        print("Loading/generating training data...")
        x_train, y_train, x_val, y_val = load_bean_data()
        
        # Build CNN model
        print("Building CNN model...")
        model = build_cnn_model()
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=config['learning_rate']),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Starting training...")
        # Train model
        history = model.fit(
            x_train, y_train,
            epochs=config['epochs'],
            batch_size=config['batch_size'],
            validation_data=(x_val, y_val),
            verbose=1
        )
        
        # Log metrics
        final_metrics = {
            'accuracy': float(history.history['accuracy'][-1]),
            'val_accuracy': float(history.history['val_accuracy'][-1]),
            'loss': float(history.history['loss'][-1]),
            'val_loss': float(history.history['val_loss'][-1]),
            'best_val_accuracy': float(max(history.history['val_accuracy']))
        }
        
        mlflow.log_metrics(final_metrics)
        
        # Save model to MLflow
        print("Saving model to MLflow...")
        mlflow.tensorflow.log_model(model, "model")
        
        # Convert to TFLite
        print("Converting to TFLite...")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = "/tmp/bean_model.tflite"
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        mlflow.log_artifact(tflite_path, "tflite_model")
        
        model_size_mb = len(tflite_model) / (1024 * 1024)
        mlflow.log_metric("tflite_size_mb", model_size_mb)
        
        print(f"Training completed successfully!")
        print(f"Final validation accuracy: {final_metrics['val_accuracy']:.4f}")
        print(f"TFLite model size: {model_size_mb:.2f} MB")
        
        # Return results for Airflow to use
        results = {
            **final_metrics,
            'mlflow_run_id': mlflow.active_run().info.run_id,
            'tflite_size_mb': model_size_mb
        }
        
        # Write results to file for Airflow to read
        with open('/tmp/training_results.json', 'w') as f:
            json.dump(results, f)
        
        return results

def load_bean_data():
    """Load bean disease dataset - replace with your actual data loading"""
    print("Generating synthetic bean disease dataset...")
    
    # Synthetic data for 3 classes: healthy, angular_leaf_spot, bean_rust
    num_samples = 1200
    x_data = np.random.rand(num_samples, 224, 224, 3)  # Random images
    y_data = np.random.randint(0, 3, num_samples)
    y_data = tf.keras.utils.to_categorical(y_data, 3)
    
    # Train/validation split
    split_idx = int(0.8 * num_samples)
    x_train, x_val = x_data[:split_idx], x_data[split_idx:]
    y_train, y_val = y_data[:split_idx], y_data[split_idx:]
    
    print(f"Dataset loaded: {len(x_train)} training, {len(x_val)} validation samples")
    return x_train, y_train, x_val, y_val

def build_cnn_model():
    """Build CNN model for bean disease classification"""
    model = tf.keras.Sequential([
        # First Conv Block
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Second Conv Block
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Third Conv Block
        tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        
        # Classifier
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes
    ])
    
    return model

if __name__ == "__main__":
    main()