import sys, time, json, math, os, pathlib
import numpy as np
import cv2
import tensorflow as tf

"""
사용법:
python tflite_test.py export_tflite/drowsy_fp16.tflite export_ckpt/labels.json input.mp4

출력:
- 콘솔: 총 프레임수, 총 실행시간, 평균 FPS, 프레임별 (idx, t_ms, label, prob)
- CSV 저장: ./video_preds.csv  (원치 않으면 관련 코드 주석 처리)
"""

def unique_csv_path(video_path, outdir="."):
    """<video_stem>.csv가 있으면 <video_stem>_1.csv, _2.csv ... 로 저장 경로를 반환"""
    stem = pathlib.Path(video_path).stem
    i = 0
    while True:
        name = f"{stem}.csv" if i == 0 else f"{stem}_{i}.csv"
        candidate = pathlib.Path(outdir) / name
        if not candidate.exists():
            return str(candidate)
        i += 1

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    # labels.json {"classes":["DROWSY","NATURAL"]} 형태 가정
    return obj["classes"]

def prepare_input(frame_bgr, input_details):
    """모델 입력 dtype/양자화에 맞춰 준비.
       - float32 입력: MobileNetV2 preprocess([-1,1]) 적용
       - uint8 입력: preprocess 후 (float -> 양자화) scale/zero_point로 변환
    """
    # 입력 텐서 정보
    iinfo = input_details[0]
    _, H, W, C = iinfo["shape"]
    dtype = iinfo["dtype"]
    quant = iinfo.get("quantization", (0.0, 0))

    # BGR->RGB, resize
    x = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (W, H))

    if dtype == np.float32:
        x = x.astype(np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # [-1,1]
        x = np.expand_dims(x, 0)
        return x.astype(np.float32)

    elif dtype == np.uint8:
        # uint8 완전정수 모델: preprocess 후 양자화
        x = x.astype(np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # float([-1,1])
        scale, zp = quant if quant is not None else (0.0, 0)
        if scale == 0.0:
            # 안전장치: 양자화 정보가 비정상인 경우, 0~255로 단순 매핑
            x = ((x + 1.0) * 127.5)
        else:
            # float -> quantized
            x = x / scale + zp
        x = np.clip(np.round(x), 0, 255).astype(np.uint8)
        x = np.expand_dims(x, 0)
        return x
    else:
        raise ValueError(f"Unsupported input dtype: {dtype}")

def dequant_output(y, oinfo):
    """양자화된 출력이면 역양자화해서 float로 변환."""
    if oinfo["dtype"] == np.uint8:
        scale, zp = oinfo.get("quantization", (0.0, 0))
        if scale == 0.0:
            y = y.astype(np.float32) / 255.0
        else:
            y = (y.astype(np.float32) - zp) * scale
        # 혹시 softmax 미세 오차가 있으면 정규화
        s = y.sum()
        if s > 0:
            y = y / s
        return y
    else:
        return y.astype(np.float32)

def main():
    if len(sys.argv) < 4:
        print("Usage: python infer_video_tflite.py <model.tflite> <labels.json> <input.mp4>")
        sys.exit(1)

    model_path = sys.argv[1]
    labels_path = sys.argv[2]
    video_path  = sys.argv[3]

    labels = load_labels(labels_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    iinfo = input_details[0]
    oinfo = output_details[0]

    # WARMUP (옵션)
    ret, frame = cap.read()
    if not ret:
        print("Empty video.")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 되감기
    _ = prepare_input(frame, input_details)  # 비용 미리 계산
    interpreter.set_tensor(iinfo["index"], prepare_input(frame, input_details))
    interpreter.invoke()

    # 추론 루프
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    per_frame_ms = []
    preds = []

    t0 = time.perf_counter()
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        inp = prepare_input(frame, input_details)
        t1 = time.perf_counter()
        interpreter.set_tensor(iinfo["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(oinfo["index"])[0]
        out = dequant_output(out, oinfo)
        t2 = time.perf_counter()

        latency_ms = (t2 - t1) * 1000.0
        per_frame_ms.append(latency_ms)

        cls = int(np.argmax(out))
        prob = float(out[cls]) if out.ndim == 1 and cls < out.shape[0] else float(np.max(out))
        preds.append((idx, latency_ms, labels[cls], prob))

        # 콘솔에 간단 표기 (원하면 끄기 가능)
        print(f"[{idx:05d}] {latency_ms:7.2f} ms  ->  {labels[cls]}  (p={prob:.3f})")
        idx += 1

    t1 = time.perf_counter()
    cap.release()

    total_time = t1 - t0
    total_frames = len(per_frame_ms)
    avg_ms = (sum(per_frame_ms)/total_frames) if total_frames else float("nan")
    med_ms = (np.median(per_frame_ms) if total_frames else float("nan"))
    fps = (total_frames / total_time) if total_time > 0 else 0.0

    print("\n===== SUMMARY =====")
    print(f"Frames processed : {total_frames}")
    print(f"Total time       : {total_time:.3f} s")
    print(f"Avg latency      : {avg_ms:.2f} ms")
    print(f"Median latency   : {med_ms:.2f} ms")
    print(f"Throughput (FPS) : {fps:.2f}")

    # CSV 저장 (원치 않으면 주석처리)
    import csv
    csv_path = unique_csv_path(video_path)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame_idx", "latency_ms", "label", "prob"])
        for r in preds:
            w.writerow(r)
    print(f'Saved per-frame results -> "{csv_path}"')

if __name__ == "__main__":
    main()
