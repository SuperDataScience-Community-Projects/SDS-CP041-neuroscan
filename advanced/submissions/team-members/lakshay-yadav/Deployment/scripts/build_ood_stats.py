import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# ------------------------------------------------------------ #
# 🧭 PATH CONFIGURATION (Lakshay – NeuroScan Project)
# ------------------------------------------------------------ #
REPO_ROOT = r"C:\Users\ABCD\_ML projects(SDS)\SDS-CP041-neuroscan\advanced\submissions\team-members\lakshay-yadav"
DEPLOYMENT_ROOT = os.path.join(REPO_ROOT, "Deployment")

# Model and OOD stats inside Deployment/assets
MODEL_PATH = os.path.join(DEPLOYMENT_ROOT, "assets", "best_efficientnetb0_fixed.keras")
OOD_STATS_PATH = os.path.join(DEPLOYMENT_ROOT, "assets", "ood_stats.npz")

# Dataset is outside Deployment, inside repo "assets" folder
DATASET_PATH = os.path.join(REPO_ROOT, "assets", "Dataset")

# Add app directory for preprocessing import
import sys
sys.path.insert(0, os.path.join(DEPLOYMENT_ROOT, "app"))
from preprocessing import preprocess_image

# ------------------------------------------------------------ #
# 🧠 Build OOD Statistics
# ------------------------------------------------------------ #
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")
if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(f"❌ Dataset directory not found at {DATASET_PATH}")

print(f"✅ Loading model from: {MODEL_PATH}")
model = load_model(MODEL_PATH)

# Penultimate feature extractor (second-last layer)
feat_model = Model(inputs=model.input, outputs=model.layers[-2].output)

datagen = ImageDataGenerator()
gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

print(f"📊 Extracting features from {len(gen.filenames)} MRI images...")
feature_batches = []
for i in range(len(gen)):
    x, _ = gen[i]
    # Convert to PIL Image and re-preprocess for model consistency
    x_proc = np.stack([preprocess_image(Image.fromarray((img * 255).astype(np.uint8))) for img in x], axis=0)
    features = feat_model.predict(x_proc, verbose=0)
    feature_batches.append(features)

features_all = np.concatenate(feature_batches, axis=0)
print(f"✅ Feature extraction complete. Shape: {features_all.shape}")

# ------------------------------------------------------------ #
# 🧮 Compute Mahalanobis OOD Statistics
# ------------------------------------------------------------ #
mean = features_all.mean(axis=0)
cov = np.cov(features_all, rowvar=False) + 1e-6 * np.eye(features_all.shape[1])
cov_inv = np.linalg.inv(cov)
d2 = np.sum((features_all - mean) @ cov_inv * (features_all - mean), axis=1)
thr = float(np.percentile(d2, 99))  # 99th percentile cutoff

# Save stats
np.savez(OOD_STATS_PATH, mean=mean, cov_inv=cov_inv, thr=thr)
print(f"✅ OOD stats saved to: {OOD_STATS_PATH}")
print(f"ℹ️ Mahalanobis 95th percentile threshold: {thr:.4f}")
