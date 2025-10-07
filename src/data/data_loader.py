"""
Data loading and preprocessing for bean disease classification
"""

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import collections

from config.training_config import TrainingConfig, DatasetSource


class BeanDataLoader:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.class_names = ["angular_leaf_spot", "bean_rust", "healthy"]
        self.n_classes = len(self.class_names)

    def load_data(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Load and split the bean disease dataset"""
        print("Loading bean disease dataset...")

        if self.config.dataset_source == DatasetSource.TENSORFLOW:
            # Load dataset directly without converting to dataframe
            ds, info = tfds.load(
                "beans",
                split="all",
                with_info=True,
                shuffle_files=False,  # We'll shuffle after splitting
                as_supervised=True,
            )

            # Convert to lists for splitting (indices only)
            images_list = []
            labels_list = []

            for image, label in ds:
                images_list.append(image.numpy())
                labels_list.append(label.numpy())

            labels_array = np.array(labels_list)

        elif self.config.dataset_source == DatasetSource.HUGGING_FACE:
            from datasets import load_dataset, concatenate_datasets
            import io
            from PIL import Image

            ds = load_dataset("AI-Lab-Makerere/beans")
            ds_all = concatenate_datasets([ds["train"], ds["validation"], ds["test"]])

            # Convert PIL images to numpy arrays
            images_list = []
            labels_list = []

            for item in ds_all:
                # Convert PIL Image to numpy array
                img = item["image"]
                if isinstance(img, Image.Image):
                    img_array = np.array(img)
                else:
                    img_array = img
                images_list.append(img_array)
                labels_list.append(item["labels"])

            labels_array = np.array(labels_list)

        print(f"Total samples: {len(images_list)}")

        # Stratified split using indices
        indices = np.arange(len(images_list))
        train_idx, temp_idx = train_test_split(
            indices,
            train_size=self.config.train_size,
            random_state=self.config.random_seed,
            shuffle=True,
            stratify=labels_array,
        )

        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=self.config.val_size,
            random_state=self.config.random_seed,
            shuffle=True,
            stratify=labels_array[temp_idx],
        )

        # Create datasets using generator to avoid loading all into memory at once
        def make_generator(indices):
            def gen():
                for idx in indices:
                    yield images_list[idx], labels_array[idx]
            return gen

        # Create TensorFlow datasets from generators
        ds_train = tf.data.Dataset.from_generator(
            make_generator(train_idx),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                tf.TensorSpec(shape=(), dtype=tf.int64),
            )
        )

        ds_valid = tf.data.Dataset.from_generator(
            make_generator(val_idx),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                tf.TensorSpec(shape=(), dtype=tf.int64),
            )
        )

        ds_test = tf.data.Dataset.from_generator(
            make_generator(test_idx),
            output_signature=(
                tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
                tf.TensorSpec(shape=(), dtype=tf.int64),
            )
        )

        print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

        # Verify class distribution
        train_labels = labels_array[train_idx]
        val_labels = labels_array[val_idx]
        test_labels = labels_array[test_idx]

        self._verify_class_distribution_from_labels(train_labels, "Training")
        self._verify_class_distribution_from_labels(val_labels, "Validation")
        self._verify_class_distribution_from_labels(test_labels, "Test")

        return ds_train, ds_valid, ds_test

    def _verify_class_distribution(self, dataset: tf.data.Dataset, split_name: str):
        """Verify class distribution is maintained"""
        label_counts = collections.Counter()
        for _, label in dataset:
            label_counts[label.numpy()] += 1

        total = sum(label_counts.values())
        ratios = {label: count / total for label, count in label_counts.items()}
        print(f"{split_name} class distribution: {ratios}")

    def _verify_class_distribution_from_labels(self, labels: np.ndarray, split_name: str):
        """Verify class distribution from label array"""
        label_counts = collections.Counter(labels)
        total = len(labels)
        ratios = {int(label): count / total for label, count in label_counts.items()}
        print(f"{split_name} class distribution: {ratios}")
