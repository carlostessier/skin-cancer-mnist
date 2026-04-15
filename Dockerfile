# GPU image — includes CUDA 12.2 + cuDNN 8 + TensorFlow 2.15
# TF auto-detects GPU at runtime; falls back to CPU if no GPU present
FROM tensorflow/tensorflow:2.15.0-gpu

LABEL maintainer="AI & Big Data Vocational Course"
LABEL description="Skin cancer classification with Keras/TensorFlow (GPU/CPU)"

WORKDIR /workspace

# System deps for image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8888

CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", \
     "--port=8888", \
     "--no-browser", \
     "--allow-root", \
     "--ServerApp.token=''", \
     "--ServerApp.password=''", \
     "--notebook-dir=/workspace"]
