# export_tflite.py
import os, json, random, numpy as np, tensorflow as tf
from pathlib import Path
from PIL import Image

DATA_ROOT = "Drowsy_Detection_Dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
OUT_DIR   = "Drowsy_Detection_Dataset/Drowsy_Detection_Dataset_export"
SAVED     = os.path.join(OUT_DIR, "saved_model")
EXPORT    = "export_tflite"
IMG_SIZE  = (224, 224)

Path(EXPORT).mkdir(parents=True, exist_ok=True)
random.seed(42)

# === Float32 ===
conv = tf.lite.TFLiteConverter.from_saved_model(SAVED)
open(os.path.join(EXPORT, "drowsy_fp32.tflite"), "wb").write(conv.convert())

# === Float16 ===
conv = tf.lite.TFLiteConverter.from_saved_model(SAVED)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
open(os.path.join(EXPORT, "drowsy_fp16.tflite"), "wb").write(conv.convert())

# === INT8 (대표 샘플: 실제 이미지 150장 사용 예) ===
def list_images(root):
    exts = {".jpg",".jpeg",".png",".bmp",".webp"}
    paths = []
    for cls in os.listdir(root):
        d = os.path.join(root, cls)
        if not os.path.isdir(d): continue
        for fn in os.listdir(d):
            if os.path.splitext(fn)[1].lower() in exts:
                paths.append(os.path.join(d, fn))
    return paths

train_imgs = list_images(TRAIN_DIR)
random.shuffle(train_imgs)
rep_paths = train_imgs[:150] if len(train_imgs) >= 150 else train_imgs

def rep_dataset_gen():
    for p in rep_paths:
        img = Image.open(p).convert("RGB").resize(IMG_SIZE)
        x = np.asarray(img).astype(np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # [-1,1]
        x = np.expand_dims(x, 0)
        yield [x]

conv = tf.lite.TFLiteConverter.from_saved_model(SAVED)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep_dataset_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# **주의**: 아래 두 줄은 완전 정수(uint8) 입출력을 강제.
# 모바일에서 속도/메모리 유리하지만, 전처리 파이프라인을 uint8로 맞출 때만 켜세요.
# conv.inference_input_type = tf.uint8
# conv.inference_output_type = tf.uint8

open(os.path.join(EXPORT, "drowsy_int8.tflite"), "wb").write(conv.convert())

print("Exported:", [p.name for p in Path(EXPORT).glob("*.tflite")])
print("Labels:", os.path.join(OUT_DIR, "labels.json"))
