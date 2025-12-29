# This was retrieved from the HuggingFace model registry: https://huggingface.co/prithivMLmods/deepfake-detector-model-v1
# GitHub: https://github.com/PRITHIVSAKTHIUR/deepfake-detector-model-v1

from transformers import AutoImageProcessor, SiglipForImageClassification
from PIL import Image
import torch
import os
from pathlib import Path
import csv

# Load model and processor
model_name = "prithivMLmods/deepfake-detector-model-v1"  
model = SiglipForImageClassification.from_pretrained(model_name)
processor = AutoImageProcessor.from_pretrained(model_name)

# Updated label mapping
id2label = {
    "0": "fake",
    "1": "real"
}

def classify_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()

    prediction = {
        id2label[str(i)]: round(probs[i], 3) for i in range(len(probs))
    }

    return prediction