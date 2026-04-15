# Skin Cancer MNIST — Clasificador con Deep Learning

Clasificador binario de lesiones cutáneas entrenado con el dataset **HAM10000**.  
Detecta si un lunar requiere consulta dermatológica usando Transfer Learning con MobileNetV2.

> Proyecto educativo — Formación Profesional en Inteligencia Artificial y Big Data

---

## ¿Qué hace?

Recibe una imagen dermatoscópica de un lunar y predice:

| Etiqueta | Significado | Clases HAM10000 |
|----------|-------------|-----------------|
| 0 — Benigno | Sin urgencia aparente | `nv`, `bkl`, `df`, `vasc` |
| 1 — Revisar | Consultar dermatólogo | `mel`, `bcc`, `akiec` |

---

## Dataset

**HAM10000 — Skin Cancer MNIST**  
10.015 imágenes dermatoscópicas de 7 tipos de lesiones cutáneas.

Fuente: [Kaggle — kmader/skin-lesion-analysis-toward-melanoma-detection](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)

El dataset **no está incluido** en el repositorio (2.5 GB). Ver sección de instalación.

---

## Arquitectura del Modelo

Transfer Learning en 2 fases sobre **MobileNetV2** (preentrenada en ImageNet):

```
Input (224 × 224 × 3)
        ↓
MobileNetV2 (backbone, pesos ImageNet)
        ↓
GlobalAveragePooling2D
        ↓
Dense(256, relu) + Dropout(0.4)
        ↓
Dense(64, relu)  + Dropout(0.2)
        ↓
Dense(1, sigmoid)   → probabilidad [0, 1]
```

**Fase 1 — Feature Extraction:** backbone congelado, solo entrena la cabeza.  
**Fase 2 — Fine-tuning:** descongelar el 30% final del backbone con LR × 0.1.

---

## Técnicas aplicadas

| Técnica | Problema que resuelve |
|---|---|
| Transfer Learning | Dataset pequeño (10k imágenes) |
| Fine-tuning | Adaptar features al dominio médico |
| Image Augmentation | Overfitting, variedad artificial |
| Class Weights | Desbalanceo 4:1 (benigno vs maligno) |
| EarlyStopping | Parar antes de overfitting |
| Umbral ajustado (0.4) | Priorizar recall en contexto médico |

---

## Estructura del Repositorio

```
skin-cancer-mnist/
├── notebooks/
│   ├── skin_cancer_classifier.ipynb      # Notebook principal (EDA → entrenamiento → evaluación)
│   ├── best_model_phase1.keras           # Mejor modelo Fase 1
│   ├── best_model_phase2.keras           # Mejor modelo Fase 2
│   ├── skin_cancer_classifier_final.keras
│   ├── skin_cancer_savedmodel/           # Formato TF SavedModel (producción)
│   └── *.png                             # Gráficas generadas
├── data/
│   └── HAM10000_metadata.csv             # Metadata del dataset (incluida)
├── Dockerfile                            # Imagen con TF 2.15 GPU
├── docker-compose.yml
├── .devcontainer/                        # Dev Container para VS Code
├── requirements.txt
├── download_dataset.sh                   # Script para descargar dataset con Kaggle API
└── test_pipeline.py
```

---

## Instalación y Uso

### Opción A — Docker (recomendado)

```bash
# 1. Clonar repositorio
git clone https://github.com/carlostessier/skin-cancer-mnist.git
cd skin-cancer-mnist

# 2. Levantar contenedor
docker-compose up --build

# 3. Descargar dataset (requiere ~/.kaggle/kaggle.json)
docker exec -it <container> bash download_dataset.sh
```

JupyterLab disponible en `http://localhost:8888`.

### Opción B — Dev Container (VS Code)

Abrir el repositorio en VS Code → "Reopen in Container".

### Opción C — Local

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab notebooks/skin_cancer_classifier.ipynb
```

### Descargar el Dataset

**Con Kaggle API:**
```bash
kaggle datasets download -d kmader/skin-lesion-analysis-toward-melanoma-detection \
    -p data --unzip
```

> Necesitas `~/.kaggle/kaggle.json` con tus credenciales.  
> Descárgalo en: kaggle.com/settings → API → Create New Token

**Manual:** Descargar desde Kaggle y descomprimir en `data/`. Estructura esperada:
```
data/
├── HAM10000_metadata.csv
├── HAM10000_images_part_1/   (5.000 imágenes .jpg)
└── HAM10000_images_part_2/   (5.015 imágenes .jpg)
```

---

## Inferencia

```python
import tensorflow as tf
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = tf.keras.models.load_model('notebooks/skin_cancer_classifier_final.keras')

def predict_mole(image_path, threshold=0.4):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    x = preprocess_input(np.expand_dims(np.array(img), 0).astype('float32'))
    prob = float(model.predict(x, verbose=0)[0][0])
    return {
        'probabilidad_maligno': prob,
        'veredicto': 'CONSULTAR DERMATÓLOGO' if prob >= threshold else 'Sin urgencia aparente'
    }

print(predict_mole('mi_lunar.jpg'))
```

---

## Métricas (conjunto de test)

Las gráficas de evaluación se generan automáticamente en el notebook:

- `class_distribution.png` — distribución de clases
- `augmentation_examples.png` — ejemplos de image augmentation
- `training_history.png` — curvas de loss, accuracy, AUC y recall
- `evaluation_metrics.png` — matriz de confusión + curva ROC

> En contexto médico se prioriza **Recall** sobre Precisión:  
> un falso negativo (maligno no detectado) es más peligroso que una alarma falsa.

---

## Limitaciones

- Modelo **educativo** — no usar para diagnóstico clínico real sin validación médica.
- Entrenado con imágenes dermatoscópicas (equipamiento especializado) — fotos de móvil pueden diferir.
- Dataset con sesgo: 80% lesiones benignas.

---

## Stack

- Python 3.11
- TensorFlow 2.15 / Keras 2.15
- MobileNetV2 (ImageNet)
- scikit-learn, pandas, matplotlib, seaborn
- JupyterLab
- Docker + GPU support
