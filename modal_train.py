"""
Train bean disease model on Modal cloud GPUs
Similar to Colab training but with on-demand cloud GPUs
"""
import modal

app = modal.App("bean-disease-training")

# Make sure you have latest Docker image on GitHub (run build_and_push_docker GH Action if not)
image = modal.Image.from_registry("ghcr.io/martinrajniak/bean-disease-classification-image:latest")

# Volume for dataset caching (so you don't re-download every time)
dataset_volume = modal.Volume.from_name("bean-dataset-cache", create_if_missing=True)

@app.function(
    gpu="T4",  # Options: "T4", "A10G", "A100-40GB", "A100-80GB"
    image=image,
    volumes={"/data": dataset_volume},
    timeout=3600,  # 1 hour max
)
def train_model(
    base_model: str = "MOBILE_NET", # Options: "MOBILE_NET", "EFFICIENT_NET_V2", "XCEPTION" 
    epochs_pretrain: int = 5,
    epochs_finetune: int = 10,
    batch_size: int = 16,
    initial_lr: float = 0.1,
    finetune_lr: float = 0.01,
):
    """Train model on Modal GPU and return model files"""
    import sys
    import os
    import tempfile
    from datetime import datetime

    # Set working directory and Python path
    # The source code expects to import from config, data, models, training directly
    # so we add /app/src to the path
    sys.path.insert(0, "/app/src")

    # Cache dataset in volume
    os.environ['TFDS_DATA_DIR'] = '/data/tensorflow_datasets'

    from config.training_config import TrainingConfig, BaseModel, Optimizer, DatasetSource
    from training.trainer import train_model_core

    # Map string to enum
    base_model_map = {
        "XCEPTION": BaseModel.XCEPTION,
        "EFFICIENT_NET_V2": BaseModel.EFFICIENT_NET_V2,
        "MOBILE_NET": BaseModel.MOBILE_NET,
    }

    # Create config
    config = TrainingConfig(
        base_model=base_model_map[base_model],
        optimizer=Optimizer.SGD,
        dataset_source=DatasetSource.TENSORFLOW,
        epochs_pretrain=epochs_pretrain,
        epochs_finetune=epochs_finetune,
        initial_lr=initial_lr,
        finetune_lr=finetune_lr,
        batch_size=batch_size,
        experiment_name="bean_disease_modal",
        run_name=f"modal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )

    print("=" * 60)
    print(f"Training on Modal with GPU: {base_model}")
    print(f"Epochs: {epochs_pretrain} + {epochs_finetune}")
    print("=" * 60)

    # Train using same code as Colab
    output_dir = tempfile.mkdtemp()
    result = train_model_core(config, output_dir=output_dir)

    # Read model files
    with open(f"{output_dir}/bean_disease_model.keras", "rb") as f:
        keras_model = f.read()

    with open(f"{output_dir}/bean_disease_model.tflite", "rb") as f:
        tflite_model = f.read()

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Test Accuracy: {result['metrics']['test_accuracy']:.4f}")
    print(f"TFLite Size:   {result['tflite_size_mb']:.2f} MB")
    print("=" * 60)

    # Commit volume to save cached dataset
    dataset_volume.commit()

    return {
        "keras_model": keras_model,
        "tflite_model": tflite_model,
        "metrics": result["metrics"],
        "config": config.to_dict(),
    }


@app.local_entrypoint()
def main(
    base_model: str = "MOBILE_NET",
    epochs_pretrain: int = 5,
    epochs_finetune: int = 10,
    gpu: str = "A10G",
):
    """
    Train model and download results locally

    Usage:
        modal run modal_train.py --base-model XCEPTION --gpu A100-40GB
    """
    from pathlib import Path
    from datetime import datetime

    print(f"🚀 Starting training on Modal with {gpu} GPU...")

    # Update GPU in decorator (dynamically)
    result = train_model.remote(
        base_model=base_model,
        epochs_pretrain=epochs_pretrain,
        epochs_finetune=epochs_finetune,
    )

    # Save models locally
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path("models") / f"modal_{base_model.lower()}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    keras_path = output_dir / "bean_disease_model.keras"
    tflite_path = output_dir / "bean_disease_model.tflite"

    with open(keras_path, "wb") as f:
        f.write(result["keras_model"])

    with open(tflite_path, "wb") as f:
        f.write(result["tflite_model"])

    print(f"\n📦 Models saved to: {output_dir}")
    print(f"   - {keras_path.name}")
    print(f"   - {tflite_path.name}")
    print(f"\n📊 Final Metrics:")
    for key, value in result["metrics"].items():
        print(f"   {key}: {value:.4f}")
