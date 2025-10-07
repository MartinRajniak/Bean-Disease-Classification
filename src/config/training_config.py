"""
Training configuration management
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional


class DatasetSource(Enum):
    HUGGING_FACE = auto()
    TENSORFLOW = auto()
    KAGGLE = auto()


class BaseModel(Enum):
    XCEPTION = auto()
    EFFICIENT_NET_V2 = auto()
    MOBILE_NET = auto()


class Optimizer(Enum):
    SGD = auto()
    ADAM = auto()
    NADAM = auto()


@dataclass
class TrainingConfig:

    # Model settings
    base_model: BaseModel = BaseModel.XCEPTION
    optimizer: Optimizer = Optimizer.SGD
    preprocess_in_model: bool = False

    # Dataset settings
    dataset_source: DatasetSource = DatasetSource.TENSORFLOW
    train_size: int = 1034
    val_size: int = 133
    test_size: int = 128

    # Training parameters
    batch_size: int = 16
    epochs_pretrain: int = 5
    epochs_finetune: int = 10

    # Learning rates
    initial_lr: float = 0.1
    finetune_lr: float = 0.01

    # Regularization
    dropout_rate: float = 0.5

    # Early stopping
    patience: int = 10

    # Reproducibility
    random_seed: int = 42

    # MLflow settings
    experiment_name: str = "bean_disease_classification"
    run_name: str = "training_unknown"
    mlflow_tracking_uri: str = "http://mlflow:5000"

    def to_dict(self) -> dict:
        """Convert to dictionary for logging"""
        return {
            "base_model": self.base_model.name,
            "optimizer": self.optimizer.name,
            "dataset_source": self.dataset_source.name,
            "batch_size": self.batch_size,
            "epochs_pretrain": self.epochs_pretrain,
            "epochs_finetune": self.epochs_finetune,
            "initial_lr": self.initial_lr,
            "finetune_lr": self.finetune_lr,
            "dropout_rate": self.dropout_rate,
            "random_seed": self.random_seed,
        }
