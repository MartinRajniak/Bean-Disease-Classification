import tensorflow as tf

# Configure GPU memory growth BEFORE any TensorFlow operations
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU setup warning: {e}")

# Set up deterministic behavior AFTER GPU configuration
tf.keras.utils.set_random_seed(42)


from config.training_config import TrainingConfig, BaseModel, Optimizer
from data.data_loader import BeanDataLoader
from models.model_builder import BeanModelBuilder


def train_model_core(config: TrainingConfig, output_dir=None):
    """
    Core training function without MLflow dependency.
    Can be used standalone or wrapped with MLflow tracking.

    Args:
        config: Training configuration
        output_dir: Optional directory to save models (if None, models not saved to disk)

    Returns:
        dict with training results including model, tflite_model, and metrics
    """
    # Load data
    print("Loading dataset...")
    data_loader = BeanDataLoader(config)
    ds_train, ds_valid, ds_test = data_loader.load_data()

    # Prepare data pipeline
    print("Preparing data pipeline...")
    ds_train, ds_valid, ds_test = prepare_data_pipeline(
        ds_train, ds_valid, ds_test, config
    )

    # Build model
    print("Building model...")
    model_builder = BeanModelBuilder(config)
    model, base_model = model_builder.build_model()

    print("Model architecture:")
    model.summary()

    # Phase 1: Initial training
    print("Phase 1: Initial training (frozen base model)...")
    optimizer = get_optimizer(config, phase="pretrain")
    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer=optimizer,
        metrics=["accuracy"],
    )

    callbacks = get_callbacks(config, phase="pretrain")
    history_pretrain = model.fit(
        ds_train,
        validation_data=ds_valid,
        epochs=config.epochs_pretrain,
        callbacks=callbacks,
        verbose=1,
    )

    # Phase 2: Fine-tuning
    if config.epochs_finetune > 0:
        print("Phase 2: Fine-tuning (unfrozen layers)...")
        model_builder.prepare_for_finetuning(model, base_model)

        optimizer = get_optimizer(config, phase="finetune")
        model.compile(
            loss="sparse_categorical_crossentropy",
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        callbacks = get_callbacks(config, phase="finetune")
        history_finetune = model.fit(
            ds_train,
            validation_data=ds_valid,
            epochs=config.epochs_finetune,
            callbacks=callbacks,
            verbose=1,
        )

        # Combine histories
        final_accuracy = history_finetune.history["accuracy"][-1]
        final_val_accuracy = history_finetune.history["val_accuracy"][-1]
    else:
        final_accuracy = history_pretrain.history["accuracy"][-1]
        final_val_accuracy = history_pretrain.history["val_accuracy"][-1]

    # Evaluate on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy = model.evaluate(ds_test, verbose=0)

    # Prepare metrics
    final_metrics = {
        "final_train_accuracy": float(final_accuracy),
        "final_val_accuracy": float(final_val_accuracy),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
    }

    # Convert to TFLite
    print("Converting to TFLite...")
    tflite_model = convert_to_tflite(model)
    model_size_mb = len(tflite_model) / (1024 * 1024)

    # Save models to disk if output_dir is provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save Keras model
        keras_path = f"{output_dir}/bean_disease_model.keras"
        model.save(keras_path)
        print(f"Keras model saved to: {keras_path}")

        # Save TFLite model
        tflite_path = f"{output_dir}/bean_disease_model.tflite"
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
        print(f"TFLite model saved to: {tflite_path}")

    result = {
        "status": "success",
        "model": model,
        "tflite_model": tflite_model,
        "metrics": final_metrics,
        "tflite_size_mb": model_size_mb,
        "training_config": config.to_dict(),
        "message": f"Training completed! Test accuracy: {test_accuracy:.4f}",
    }

    print(f"Training completed successfully!")
    print(f"  Test accuracy: {test_accuracy:.4f}")
    print(f"  Test loss: {test_loss:.4f}")
    print(f"  TFLite size: {model_size_mb:.2f} MB")

    return result


def prepare_data_pipeline(ds_train, ds_valid, ds_test, config):
    """Prepare optimized data pipeline"""
    model_builder = BeanModelBuilder(config)
    preprocess, augmentation, _ = model_builder.build_preprocessing_layers()

    if not config.preprocess_in_model:
        # Apply preprocessing
        ds_train = ds_train.map(
            lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_valid = ds_valid.map(
            lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
        ds_test = ds_test.map(
            lambda x, y: (preprocess(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )

    # Training pipeline with augmentation
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(config.train_size)
    if not config.preprocess_in_model:
        ds_train = ds_train.map(
            lambda x, y: (augmentation(x), y), num_parallel_calls=tf.data.AUTOTUNE
        )
    ds_train = ds_train.batch(config.batch_size)
    ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

    # Validation pipeline
    ds_valid = ds_valid.cache()
    ds_valid = ds_valid.batch(config.batch_size)
    ds_valid = ds_valid.prefetch(tf.data.AUTOTUNE)

    # Test pipeline
    ds_test = ds_test.cache()
    ds_test = ds_test.batch(config.batch_size)
    ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_valid, ds_test


def get_optimizer(config, phase="pretrain"):
    """Get optimizer for training phase"""
    if config.optimizer == Optimizer.SGD:
        lr = config.initial_lr if phase == "pretrain" else config.finetune_lr
        return tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)
    elif config.optimizer == Optimizer.ADAM:
        lr = config.initial_lr if phase == "pretrain" else config.finetune_lr
        return tf.keras.optimizers.Adam(learning_rate=lr)
    elif config.optimizer == Optimizer.NADAM:
        lr = 1e-4 if phase == "pretrain" else 1e-5
        return tf.keras.optimizers.Nadam(learning_rate=lr)


def get_callbacks(config, phase="pretrain"):
    """Get callbacks for training phase"""
    callbacks = []

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        restore_best_weights=True,
        mode="min",
    )
    callbacks.append(early_stopping)

    return callbacks


def convert_to_tflite(model):
    """Convert Keras model to TFLite"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimizations for mobile deployment
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float32]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32

    # Enable Select TF Ops for complex models like Xception
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]

    return converter.convert()
