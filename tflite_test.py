import sys, time, json, os, pathlib, argparse
import numpy as np
import cv2
import tensorflow as tf

"""
사용법:
# (1) 동영상 파일
python tflite_test.py export_tflite/drowsy_fp16.tflite export_ckpt/labels.json input.mp4 --show

# (2) 웹캠(0번)
python tflite_test.py export_tflite/drowsy_fp16.tflite export_ckpt/labels.json --camera 0 --show --flip

옵션:
--decision-thr 0.8      # p(DROWSY) >= thr 이면 DROWSY
--drowsy-label DROWSY   # DROWSY 라벨명
--no-csv                # CSV 저장 안함
--threads N             # TFLite 스레드 수
"""

def unique_csv_path(stem_name: str, outdir: pathlib.Path) -> str:
    """outdir/stem.csv, 중복 시 stem_1.csv … 로 반환"""
    outdir.mkdir(parents=True, exist_ok=True)
    i = 0
    while True:
        name = f"{stem_name}.csv" if i == 0 else f"{stem_name}_{i}.csv"
        candidate = outdir / name
        if not candidate.exists():
            return str(candidate)
        i += 1

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["classes"]

def prepare_input(frame_bgr, input_details):
    iinfo = input_details[0]
    _, H, W, C = iinfo["shape"]
    dtype = iinfo["dtype"]
    quant = iinfo.get("quantization", (0.0, 0))

    x = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (W, H))

    if dtype == np.float32:
        x = x.astype(np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # [-1,1]
        x = np.expand_dims(x, 0)
        return x.astype(np.float32)

    elif dtype == np.uint8:
        x = x.astype(np.float32)
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)  # [-1,1]
        scale, zp = quant if quant is not None else (0.0, 0)
        if scale == 0.0:
            x = ((x + 1.0) * 127.5)
        else:
            x = x / scale + zp
        x = np.clip(np.round(x), 0, 255).astype(np.uint8)
        x = np.expand_dims(x, 0)
        return x
    else:
        raise ValueError(f"Unsupported input dtype: {dtype}")

def dequant_output(y, oinfo):
    if oinfo["dtype"] == np.uint8:
        scale, zp = oinfo.get("quantization", (0.0, 0))
        if scale == 0.0:
            y = y.astype(np.float32) / 255.0
        else:
            y = (y.astype(np.float32) - zp) * scale
        s = float(y.sum())
        if s > 0:
            y = y / s
        return y.astype(np.float32)
    else:
        return y.astype(np.float32)

def open_capture(video_path, cam_index):
    """파일 경로 또는 웹캠 인덱스 중 하나를 받아 VideoCapture 생성"""
    if cam_index is not None:
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW if os.name == "nt" else 0)
        src_name = f"camera{cam_index}"
    else:
        cap = cv2.VideoCapture(video_path)
        src_name = str(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open source: {src_name}")
    return cap, src_name

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("model", help="path to .tflite")
    p.add_argument("labels", help="path to labels.json")
    p.add_argument("input", nargs="?", default=None, help="video file (omit when using --camera)")
    p.add_argument("--camera", type=int, default=None, help="webcam index (e.g., 0)")
    p.add_argument("--show", action="store_true", help="show live window with overlay")
    p.add_argument("--flip", action="store_true", help="horizontal flip (selfie view)")
    p.add_argument("--threads", type=int, default=None, help="tflite inference threads")
    p.add_argument("--no-csv", action="store_true", help="do not save CSV")
    # 결정 경계 관련 옵션
    p.add_argument("--decision-thr", type=float, default=0.8,
                   help="p(DROWSY) >= thr 이면 DROWSY, 아니면 NATURAL (default: 0.8)")
    p.add_argument("--drowsy-label", default="DROWSY",
                   help="Drowsy 라벨명 (labels.json과 일치해야 함)")
    return p.parse_args()

def main():
    args = parse_args()
    if args.camera is None and args.input is None:
        print("Usage: python tflite_test.py <model.tflite> <labels.json> <input.mp4> [--show]\n"
              "   or: python tflite_test.py <model.tflite> <labels.json> --camera 0 [--show]")
        sys.exit(1)

    labels = load_labels(args.labels)

    # Interpreter
    if args.threads is not None:
        interpreter = tf.lite.Interpreter(model_path=args.model, num_threads=args.threads)
    else:
        interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    iinfo = input_details[0]
    oinfo = output_details[0]

    cap, src_name = open_capture(args.input, args.camera)

    # WARMUP
    ret, frame = cap.read()
    if not ret:
        print("Empty stream.")
        return
    if args.flip:
        frame = cv2.flip(frame, 1)
    interpreter.set_tensor(iinfo["index"], prepare_input(frame, input_details))
    interpreter.invoke()

    # DROWSY/NATURAL 인덱스
    try:
        d_idx = labels.index(args.drowsy_label)
    except ValueError:
        d_idx = 0  # 못찾으면 0번으로 가정
    # NATURAL 인덱스(2클래스면 반대편)
    try:
        nat_idx = labels.index("NATURAL")
    except ValueError:
        nat_idx = 1 - d_idx if len(labels) == 2 else None

    # Loop
    per_frame_ms = []
    preds = []
    total_frames = 0
    t0 = time.perf_counter()
    t_last_disp = t0
    smoothed_fps = None

    while True:
        ret, frame = cap.read()
        if not ret:
            if args.camera is not None:
                continue
            else:
                break

        if args.flip:
            frame = cv2.flip(frame, 1)

        inp = prepare_input(frame, input_details)
        t1 = time.perf_counter()
        interpreter.set_tensor(iinfo["index"], inp)
        interpreter.invoke()
        out = interpreter.get_tensor(oinfo["index"])[0]
        out = dequant_output(out, oinfo)
        t2 = time.perf_counter()

        latency_ms = (t2 - t1) * 1000.0
        per_frame_ms.append(latency_ms)

        # === 결정 경계 적용 + 확률 기록 방식 ===
        # 1) p(DROWSY) 계산
        d_p = float(out[d_idx]) if d_idx < len(out) else 0.0

        # 2) 라벨 결정: p(DROWSY) >= thr ? DROWSY : NATURAL(또는 최대 기타)
        if d_p >= args.decision_thr:
            pred_label = labels[d_idx]
        else:
            if nat_idx is not None and nat_idx < len(out):
                pred_label = labels[nat_idx]
            else:
                other_idx = int(np.argmax([out[i] if i != d_idx else -1 for i in range(len(out))]))
                pred_label = labels[other_idx]

        # 3) CSV/표시에 쓰는 prob는 항상 p(DROWSY)
        pred_prob = d_p

        preds.append((total_frames, latency_ms, pred_label, pred_prob))

        # 콘솔 로그
        print(f"[{total_frames:05d}] {latency_ms:7.2f} ms  ->  {pred_label}  (p_drowsy={pred_prob:.3f})")
        total_frames += 1

        # 화면 표시
        if args.show:
            now = time.perf_counter()
            inst_fps = 1.0 / (now - t_last_disp) if (now - t_last_disp) > 0 else 0.0
            t_last_disp = now
            smoothed_fps = inst_fps if smoothed_fps is None else (0.9 * smoothed_fps + 0.1 * inst_fps)

            overlay = frame.copy()
            text = f"{pred_label}  pD={pred_prob:.2f}   {latency_ms:.1f} ms   {smoothed_fps:.1f} FPS"
            cv2.putText(overlay, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("TFLite Real-Time", overlay)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

    t1 = time.perf_counter()
    total_time = t1 - t0
    avg_ms = (sum(per_frame_ms)/len(per_frame_ms)) if per_frame_ms else float("nan")
    med_ms = (np.median(per_frame_ms) if per_frame_ms else float("nan"))
    fps = (total_frames / total_time) if total_time > 0 else 0.0

    print("\n===== SUMMARY =====")
    print(f"Frames processed : {total_frames}")
    print(f"Total time       : {total_time:.3f} s")
    print(f"Avg latency      : {avg_ms:.2f} ms")
    print(f"Median latency   : {med_ms:.2f} ms")
    print(f"Throughput (FPS) : {fps:.2f}")

    # === CSV 저장: 항상 현재 작업 폴더의 result/ 아래 ===
    if not args.no_csv:
        import csv
        result_dir = pathlib.Path("result")
        stem_name = pathlib.Path(src_name).stem if src_name else "output"
        csv_path = unique_csv_path(stem_name, result_dir)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "latency_ms", "label", "prob"])  # prob = p(DROWSY)
            for r in preds:
                w.writerow(r)
        print(f'Saved per-frame results -> "{csv_path}"')

if __name__ == "__main__":
    main()
