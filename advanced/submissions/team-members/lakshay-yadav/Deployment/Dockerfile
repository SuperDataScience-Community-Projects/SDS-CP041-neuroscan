# ==============================================================
# 🧠 NeuroScan - Brain Tumor Detection (FastAPI + TensorFlow)
# Dockerfile for Production Deployment using requirements.txt
# ==============================================================

# 1️⃣ Base image — lightweight, stable, Python 3.10
FROM python:3.10-slim

# 2️⃣ Set working directory
WORKDIR /app

# 3️⃣ Copy project files
COPY . .

# 4️⃣ Install system dependencies for Pillow, OpenCV, TensorFlow
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxrender1 libxext6 libhdf5-dev g++ \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# 5️⃣ Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# 6️⃣ Expose FastAPI port
EXPOSE 8000

# 7️⃣ Environment setup
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# 8️⃣ Start the FastAPI app with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
