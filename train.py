# train.py
import os, json, tensorflow as tf
from tensorflow.keras import layers, models

AUTOTUNE = tf.data.AUTOTUNE

# === 경로만 본인 환경에 맞게 ===
DATA_ROOT = "Drowsy_Detection_Dataset"   # 여기에 train/, test/가 들어있음
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "test")  # Kaggle 폴더에서 'test'를 검증셋으로 사용
OUT_DIR   = "Drowsy_Detection_Dataset/Drowsy_Detection_Dataset_export"

IMG_SIZE = (224, 224)
BATCH    = 32
EPOCHS_HEAD = 12    # 헤드 학습
EPOCHS_FT   = 6     # 미세튜닝

os.makedirs(OUT_DIR, exist_ok=True)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    TRAIN_DIR, label_mode="categorical",
    image_size=IMG_SIZE, batch_size=BATCH, shuffle=True
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR, label_mode="categorical",
    image_size=IMG_SIZE, batch_size=BATCH, shuffle=False
)

# 클래스 이름 저장 (알파벳 순서. 예: ['DROWSY', 'NATURAL'])
class_names = train_ds.class_names
with open(os.path.join(OUT_DIR,"labels.json"), "w", encoding="utf-8") as f:
    json.dump({"classes": class_names}, f, ensure_ascii=False, indent=2)

# 증강 + 전처리
data_augment = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.05),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

def prep(ds, training=True):
    ds = ds.map(lambda x,y: (tf.image.resize(x, IMG_SIZE), y), num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.map(lambda x,y: (data_augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda x,y: (preprocess(x), y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)

train_ds2 = prep(train_ds, training=True)
val_ds2   = prep(val_ds,   training=False)

# 모델
base = tf.keras.applications.MobileNetV2(input_shape=IMG_SIZE+(3,), include_top=False, weights="imagenet")
base.trainable = False  # 먼저 헤드만
inputs  = layers.Input(shape=IMG_SIZE+(3,))
x = preprocess(inputs)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

ckpt = tf.keras.callbacks.ModelCheckpoint(os.path.join(OUT_DIR,"best.keras"),
                                          monitor="val_accuracy", save_best_only=True)
es = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

print(">> Train head...")
model.fit(train_ds2, validation_data=val_ds2, epochs=EPOCHS_HEAD, callbacks=[ckpt, es])

# 미세튜닝 (상위 블록만 풀어도 됨)
base.trainable = True
for layer in base.layers[:-40]:  # 마지막 40개 레이어만 푼다
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
print(">> Fine-tune backbone...")
model.fit(train_ds2, validation_data=val_ds2, epochs=EPOCHS_FT, callbacks=[ckpt, es])

# SavedModel 저장
saved = os.path.join(OUT_DIR, "saved_model")   # 디렉터리 경로
model.export(saved)  # Keras 3: SavedModel 내보내기 (TFLite에 사용)
print("SavedModel exported to:", saved)
print("Labels:", os.path.join(OUT_DIR, "labels.json"))