from typing import Dict, List, Any
from PIL import Image

import os
import json
import numpy as np
import keras


class PreTrainedPipeline():
    def __init__(self, path=""):
        self.model = keras.saving.load_model(os.path.join(path, "beans_disease_classification_transfer_learning.keras"))
        with open(os.path.join(path, "config.json")) as config:
            config = json.load(config)
        self.id2label = config["id2label"]
    
    def __call__(self, inputs: "Image.Image") -> List[Dict[str, Any]]:
        preds = self.model.predict(np.array(inputs))
        preds = preds.tolist()
        labels = [
            {"label": str(self.id2label["0"]), "score": preds[0]},
            {"label": str(self.id2label["1"]), "score": preds[1]},
            {"label": str(self.id2label["2"]), "score": preds[2]},
        ]
        return labels