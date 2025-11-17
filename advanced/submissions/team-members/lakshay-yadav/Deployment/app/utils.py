"""
utils.py — helper functions for prediction interpretation
"""
import numpy as np

CLASS_MAPPING = {
    0: "No Tumor Detected",
    1: "Brain Tumor Detected"
}

def decode_prediction(prediction):
    """Convert model output to label and confidence score with smart checks."""
    pred_prob = float(prediction[0][0])

    # Handle abnormal predictions
    if np.isnan(pred_prob) or pred_prob < 0.0 or pred_prob > 1.0:
        return "Invalid MRI Image", 0.0

    label = CLASS_MAPPING[1] if pred_prob >= 0.5 else CLASS_MAPPING[0]
    confidence = round(pred_prob * 100, 2) if label == CLASS_MAPPING[1] else round((1 - pred_prob) * 100, 2)

    # Reject unrealistic / uncertain results
    if confidence < 55:
        return "Invalid MRI Image", confidence

    return label, float(confidence)
