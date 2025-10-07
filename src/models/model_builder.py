"""
Model building utilities for bean disease classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    Input,
    Resizing,
    Lambda,
    RandomFlip,
    RandomRotation,
    RandomContrast,
)
from tensorflow.keras.models import Sequential, Model

from config.training_config import TrainingConfig, BaseModel


class BeanModelBuilder:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.n_classes = 3

    def build_preprocessing_layers(self) -> Sequential:
        """Build preprocessing and augmentation layers"""
        if self.config.base_model == BaseModel.XCEPTION:
            from tensorflow.keras.applications.xception import preprocess_input
        elif self.config.base_model == BaseModel.EFFICIENT_NET_V2:
            from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
        elif self.config.base_model == BaseModel.MOBILE_NET:
            from tensorflow.keras.applications.mobilenet import preprocess_input

        # Base preprocessing
        preprocess = Sequential(
            [
                Lambda(lambda x: tf.cast(x, tf.float32)),
                Resizing(height=224, width=224, crop_to_aspect_ratio=True),
                Lambda(preprocess_input),
            ],
            name="preprocessing",
        )

        # Augmentation layers
        augmentation = Sequential(
            [
                RandomFlip(mode="horizontal", seed=self.config.random_seed),
                RandomRotation(factor=0.05, seed=self.config.random_seed),
                RandomContrast(factor=0.2, seed=self.config.random_seed),
            ],
            name="augmentation",
        )

        # Combined preprocessing and augmentation
        preprocess_and_augmentation = Sequential(
            [
                Lambda(lambda x: tf.cast(x, tf.float32)),
                Resizing(height=224, width=224, crop_to_aspect_ratio=True),
                RandomFlip(mode="horizontal", seed=self.config.random_seed),
                RandomRotation(factor=0.05, seed=self.config.random_seed),
                RandomContrast(factor=0.2, seed=self.config.random_seed),
                Lambda(preprocess_input),
            ],
            name="preprocess_and_augmentation",
        )

        return preprocess, augmentation, preprocess_and_augmentation

    def build_base_model(self) -> keras.Model:
        """Build the base model with pretrained weights"""
        if self.config.base_model == BaseModel.XCEPTION:
            from tensorflow.keras.applications.xception import Xception

            base_model = Xception(
                input_shape=(224, 224, 3), include_top=False, weights="imagenet"
            )
        elif self.config.base_model == BaseModel.EFFICIENT_NET_V2:
            from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S

            base_model = EfficientNetV2S(
                input_shape=(224, 224, 3), include_top=False, weights="imagenet"
            )
        elif self.config.base_model == BaseModel.MOBILE_NET:
            from tensorflow.keras.applications.mobilenet import MobileNet

            base_model = MobileNet(
                input_shape=(224, 224, 3), include_top=False, weights="imagenet"
            )

        return base_model

    def build_model(self) -> keras.Model:
        """Build the complete model"""
        base_model = self.build_base_model()

        if self.config.preprocess_in_model:
            _, _, preprocess_and_augmentation = self.build_preprocessing_layers()

            inputs = Input(shape=(None, None, 3))
            x = preprocess_and_augmentation(inputs)
            x = base_model(x, training=False)
        else:
            inputs = base_model.input
            x = base_model.output

        # Classification head
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.config.dropout_rate)(x)
        outputs = Dense(self.n_classes, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs, name="bean_disease_classifier")

        # Freeze base model initially
        if self.config.preprocess_in_model:
            base_model.trainable = False
        else:
            for layer in base_model.layers:
                layer.trainable = False

        return model, base_model

    def prepare_for_finetuning(self, model: keras.Model, base_model: keras.Model):
        """Prepare model for fine-tuning by unfreezing layers"""
        if self.config.preprocess_in_model:
            base_model.trainable = True
            # Freeze early layers (adjust numbers based on your experiments)
            if self.config.base_model == BaseModel.XCEPTION:
                for layer in base_model.layers[:-226]:
                    layer.trainable = False
        else:
            # Unfreeze later layers for fine-tuning
            if self.config.base_model == BaseModel.XCEPTION:
                for layer in base_model.layers[56:]:
                    layer.trainable = True

        print(
            f"Trainable layers: {sum(1 for layer in model.layers if layer.trainable)}"
        )
        print(
            f"Non-trainable layers: {sum(1 for layer in model.layers if not layer.trainable)}"
        )
