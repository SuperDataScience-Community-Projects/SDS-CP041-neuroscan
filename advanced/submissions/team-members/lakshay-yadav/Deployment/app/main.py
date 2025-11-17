import os
import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model, Model
from PIL import Image
from pathlib import Path

from app.preprocessing import preprocess_image
from app.utils import decode_prediction
from app.validator import basic_visual_checks, MahalanobisGate

# ============================================================== #
# 🧠 NeuroScan API - Brain Tumor Detection
# Version: 2.4.1 (Validated + Adaptive OOD Gate)
# ============================================================== #

app = FastAPI(
    title="🧠 NeuroScan API",
    description="Deep Learning API for Brain Tumor Detection using MRI images — AI Lab Playground Edition",
    version="2.4.1"
)

# -------------------------------------------------------------- #
# ⚙️ Static & Template Setup
# -------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# -------------------------------------------------------------- #
# 🧩 Model + Feature Extractor + OOD Gate
# -------------------------------------------------------------- #
MODEL_PATH = BASE_DIR.parent / "assets" / "best_efficientnetb0_fixed.keras"
OOD_STATS_PATH = BASE_DIR.parent / "assets" / "ood_stats.npz"

try:
    model = load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"❌ Model loading failed: {e}")

feature_model, ood_gate = None, None
try:
    try:
        feature_model = Model(inputs=model.input, outputs=model.get_layer("avg_pool").output)
    except:
        feature_model = Model(inputs=model.input, outputs=model.layers[-2].output)

    stats = np.load(OOD_STATS_PATH)
    thr_scaled = float(stats["thr"]) * 3.0  # relaxed scaling factor
    ood_gate = MahalanobisGate(stats["mean"], stats["cov_inv"], thr_scaled)
    print(f"✅ OOD gate initialized (scaled threshold: {thr_scaled:.2f})")
except Exception as e:
    print(f"⚠️ OOD gate unavailable: {e}")
    feature_model, ood_gate = None, None

# -------------------------------------------------------------- #
# 🌐 Root Endpoint
# -------------------------------------------------------------- #
@app.get("/", response_class=HTMLResponse)
def serve_ui(request: Request):
    """Serves the NeuroScan AI Lab Playground web interface."""
    return templates.TemplateResponse("index.html", {"request": request})

# -------------------------------------------------------------- #
# 🔍 Prediction Endpoint (Smart + Adaptive OOD Validation)
# -------------------------------------------------------------- #
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an MRI image (JPG/PNG), validates it, preprocesses it,
    performs model inference, and returns structured results.
    """
    try:
        # 1️⃣ File type check
        if not file.filename.lower().endswith((".jpg", ".jpeg", ".png")):
            return JSONResponse(status_code=400, content={"error": "Invalid file type. Please upload a JPG or PNG image."})

        # 2️⃣ Load image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        img_np = np.array(image)
        width, height = image.size
        brightness = float(np.mean(img_np))
        std_channels = np.std(img_np, axis=(0, 1))

        # 3️⃣ Basic heuristics
        if width < 128 or height < 128:
            return JSONResponse(status_code=400, content={"error": "Image too small. Please upload a proper MRI scan."})
        if float(np.max(std_channels) - np.min(std_channels)) > 25:
            return JSONResponse(status_code=400, content={"error": "Please upload a valid MRI brain scan (grayscale/medical type)."})
        if brightness > 240 or brightness < 5:
            return JSONResponse(status_code=400, content={"error": "Invalid image brightness — please upload a proper MRI scan."})

        # 4️⃣ MRI visual validator
        ok, info = basic_visual_checks(img_np)
        if not ok:
            return JSONResponse(status_code=400, content={"error": f"⚠️ {info}"})

        # 5️⃣ Preprocessing for model
        img_array = preprocess_image(image)
        img_array = np.expand_dims(img_array, axis=0)

        # 6️⃣ Adaptive OOD validation
        if feature_model is not None and ood_gate is not None:
            z = feature_model.predict(img_array, verbose=0)[0]
            _, d2 = ood_gate.is_in_distribution(z)

            d2_log = np.log1p(d2)
            thr_log = np.log1p(ood_gate.thr)
            ratio = d2_log / thr_log

            print(f"[DEBUG] Raw Mahalanobis distance: {d2:.2f}, log(d2): {d2_log:.2f}, log(thr): {thr_log:.2f}, ratio: {ratio:.2f}")

            # Reject only extreme OODs
            if ratio > 3.0:  # allow 3x safety margin
                return JSONResponse(
                    status_code=400,
                    content={"error": "⚠️ Invalid MRI scan (out-of-distribution image)."}
                )

        # 7️⃣ Model inference
        prediction = model.predict(img_array)
        label, confidence = decode_prediction(prediction)

        # 8️⃣ Guard invalid labels
        if label == "Invalid MRI Image":
            return JSONResponse(status_code=400, content={"error": "⚠️ Invalid MRI scan detected. Please upload a valid brain MRI image."})

        # ✅ Response
        return JSONResponse(
            content={
                "filename": file.filename,
                "prediction": label,
                "confidence": float(f"{confidence:.2f}")
            }
        )

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})
