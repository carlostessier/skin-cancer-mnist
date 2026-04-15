import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PASS = []
FAIL = []

def check(label, fn):
    try:
        result = fn()
        msg = f"  OK  {label}" + (f" — {result}" if result else "")
        print(msg)
        PASS.append(label)
    except Exception as e:
        print(f"  FAIL {label}: {e}")
        FAIL.append(label)

# ── 1. Imports ──────────────────────────────────────────────────────────────
print("\n=== 1. Imports ===")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
check("numpy / pandas / matplotlib / seaborn / PIL / sklearn", lambda: None)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
keras_ver = getattr(keras, '__version__', None) or getattr(tf, '__version__', 'unknown')
check("TensorFlow + Keras", lambda: f"TF {tf.__version__} | Keras {keras_ver}")

# ── 2. Model build ───────────────────────────────────────────────────────────
print("\n=== 2. Model build (MobileNetV2 + custom head) ===")

def build_model():
    base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base.trainable = False
    inp = keras.Input(shape=(224, 224, 3))
    x = base(inp, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    m = Model(inp, out, name='skin_cancer_classifier')
    m.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.Precision(name='precision')]
    )
    return m, base

model, base_model = build_model()
total     = model.count_params()
trainable = sum(p.numpy().size for p in model.trainable_variables)
frozen    = sum(p.numpy().size for p in model.non_trainable_variables)
check("Model build", lambda: f"total={total:,} | trainable={trainable:,} | frozen={frozen:,}")

# ── 3. Inference ─────────────────────────────────────────────────────────────
print("\n=== 3. Inference on synthetic image ===")

dummy = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
arr   = preprocess_input(np.expand_dims(dummy.astype('float32'), 0))
prob  = float(model.predict(arr, verbose=0)[0][0])
label = 'revisar' if prob >= 0.4 else 'benigno'
check("Inference", lambda: f"prob_maligno={prob:.4f} → {label}")

# ── 4. Training smoke-test ───────────────────────────────────────────────────
print("\n=== 4. Training smoke-test (synthetic data, 2 epochs) ===")

X = np.random.rand(32, 224, 224, 3).astype('float32')
y = np.random.randint(0, 2, 32).astype('float32')
h = model.fit(X, y, epochs=2, batch_size=16, verbose=0)
check("Training", lambda: f"loss={h.history['loss'][-1]:.4f} | auc={h.history['auc'][-1]:.4f}")

# ── 5. Class weights ─────────────────────────────────────────────────────────
print("\n=== 5. Class weights (imbalance handling) ===")

labels_fake = np.array([0]*700 + [1]*300)
w = compute_class_weight('balanced', classes=np.array([0, 1]), y=labels_fake)
cw = {0: w[0], 1: w[1]}
check("Class weights", lambda: f"benigno={cw[0]:.3f} | revisar={cw[1]:.3f}")

# ── 6. ImageDataGenerator ────────────────────────────────────────────────────
print("\n=== 6. ImageDataGenerator (augmentation) ===")

gen   = ImageDataGenerator(preprocessing_function=preprocess_input,
                           rotation_range=20, horizontal_flip=True, vertical_flip=True)
batch = next(gen.flow(np.random.rand(4, 224, 224, 3).astype('float32'), batch_size=4))
check("Augmentation batch", lambda: f"shape={batch.shape}")

# ── 7. Save / load model ─────────────────────────────────────────────────────
print("\n=== 7. Model save / load ===")

save_path = '/tmp/smoke_test.keras'
model.save(save_path)
loaded = keras.models.load_model(save_path)
prob2  = float(loaded.predict(arr, verbose=0)[0][0])
os.remove(save_path)
check("Save/load", lambda: f"prob_before={prob:.6f} | prob_after={prob2:.6f} | match={abs(prob-prob2)<1e-5}")

# ── 8. Fine-tune phase simulation ────────────────────────────────────────────
print("\n=== 8. Fine-tune phase (unfreeze last 30% of backbone) ===")

fine_tune_at = int(len(base_model.layers) * 0.70)
base_model.trainable = True
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy', metrics=['accuracy'])
trainable2 = sum(p.numpy().size for p in model.trainable_variables)
check("Fine-tune compile", lambda: f"trainable params now={trainable2:,} (was {trainable:,})")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "="*45)
print(f"  PASSED: {len(PASS)}/{len(PASS)+len(FAIL)} tests")
if FAIL:
    print(f"  FAILED: {FAIL}")
else:
    print("  ALL TESTS PASSED — notebook ready to use")
print("="*45 + "\n")
sys.exit(0 if not FAIL else 1)
