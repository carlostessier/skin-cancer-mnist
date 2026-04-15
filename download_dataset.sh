#!/usr/bin/env bash
# Descarga HAM10000 dentro del contenedor Docker
# Requiere ~/.kaggle/kaggle.json con tus credenciales

set -e

echo "=== Descargando HAM10000 (Skin Cancer MNIST) ==="
echo "Tamaño: ~2.5 GB — puede tardar varios minutos"

kaggle datasets download \
    -d kmader/skin-lesion-analysis-toward-melanoma-detection \
    -p /workspace/data \
    --unzip

echo ""
echo "Dataset descargado en /workspace/data/"
echo "Archivos:"
ls -lh /workspace/data/*.csv
ls /workspace/data/ | grep HAM10000_images
