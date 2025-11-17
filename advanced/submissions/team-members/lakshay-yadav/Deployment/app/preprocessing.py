"""
preprocessing.py — handles image preprocessing before feeding to model
"""

import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# Set model input size (EfficientNetB0 expects 224x224)
TARGET_SIZE = (224, 224)

def preprocess_image(image_input):
    """
    Load and preprocess an image for model prediction.
    Accepts either a Pillow Image or a byte stream.
    """
    # ✅ Handle both file-like objects and already opened images
    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    image = image.resize(TARGET_SIZE)
    img_array = np.array(image)
    img_array = preprocess_input(img_array)  # EfficientNet preprocessing
    return img_array