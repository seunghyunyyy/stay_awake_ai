# Stay Awake AI 😴➡️🟢

스마트폰/웹캠의 얼굴 정보를 활용해 **졸음 상태를 감지**하는 파이프라인입니다. 모델 **학습**, **TFLite 내보내기**, **실시간 추론(동영상/웹캠)**, **로그 분석**을 지원합니다.

> 이 리포는 `requirements.txt`의 스택(예: TensorFlow 2.20, OpenCV 4.12, PyTorch 2.8 CUDA 12.x)을 기준으로 합니다. CPU 전용/서버 환경에 맞게 조정하세요.

---

## ✨ 기능

- **학습(Training)**: 얼굴 단서 기반 졸음 분류 모델 학습
- **추론(Inference)**:
  - TFLite로 **웹캠/동영상** 실시간 추론
  - 임계치/라벨 매핑/스무딩 등 옵션
- **분석(Analysis)**: 예측 CSV 로그를 불러 간단 통계/시각화

---

## 📚 데이터셋

본 프로젝트는 Kaggle의 **Drowsy Detection Dataset**(작성자: *Yashar Jebraeily*)을 사용합니다.

- 데이터셋: https://www.kaggle.com/datasets/yasharjebraeily/drowsy-detection-dataset

> 데이터셋의 라이선스/사용 조건을 준수하세요. 결과물을 공개할 경우, 데이터셋 출처를 명시해 주세요.

---

## 📦 환경 준비

### 1) 가상환경 만들기
```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows
# .venv\Scripts\activate
```

### 2) 의존성 설치
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
**참고**
- 주요 패키지(예): TensorFlow 2.20.x(+`tf_keras`), OpenCV 4.12, Pandas/Matplotlib,  
  PyTorch 2.8.0(+CUDA 12.9), torchvision 0.23.0(+CUDA 12.9), NVIDIA CUDA 12.x 사용자 공간 라이브러리
- **CPU 전용** 환경이라면 CPU 휠(`torch==2.8.0+cpu` 등)을 설치하거나 CUDA 관련 항목을 조정하세요.

---

## 🗂️ 프로젝트 구조(예시)

```
stay_awake_ai/
├─ dataset/                 # 데이터셋 루트(train/val/test 등)
├─ export/                  # 학습 산출물(체크포인트/로그)
├─ export_tflite/           # TFLite 산출물(.tflite, labels.json)
├─ train.py                 # 학습 스크립트
├─ export_tflite.py         # TFLite 변환 스크립트
├─ tflite_test.py           # TFLite 추론(동영상/웹캠)
├─ analyze.py               # CSV 로그 분석/시각화
└─ requirements.txt
```

---

## 🚀 빠른 시작

### A) **TFLite**로 실시간/동영상 추론
```bash
# 동영상 파일
python tflite_test.py export_tflite/drowsy_fp16.tflite export_ckpt/labels.json input.mp4 --show

# 웹캠(0번) + 미러 프리뷰
python tflite_test.py export_tflite/drowsy_fp16.tflite export_ckpt/labels.json --camera 0 --show --flip
```
자주 쓰는 옵션:
- `--decision-thr 0.8` → p≥0.8이면 DROWSY로 판정  
- `--drowsy-label DROWSY` → 양성 라벨명 지정  
- `--no-csv` → CSV 저장 비활성화

### B) 예측 로그 분석
```bash
python analyze.py ./result/Non_Drowsy.csv --plot
```

### C) 모델 학습
```bash
python train.py \
  --data-root ./dataset \
  --epochs 20 \
  --batch-size 32 \
  --img-size 224 \
  --out ./export
```

### D) **TFLite 변환**
```bash
python export_tflite.py \
  --ckpt ./export/best_model.h5 \
  --out ./export_tflite/drowsy_fp16.tflite \
  --quantize fp16
```

> 각 스크립트의 실제 인자 목록은 `--help`로 확인하세요.

---

## 🔧 트러블슈팅

- **웹캠 미검출**: `--camera 0/1/2…` 인덱스 변경  
- **FPS 저하**: 입력 해상도 축소, FP16/INT8 양자화, 프레임 스킵 적용  
- **라벨/전처리 불일치**: 학습/추론의 리사이즈·정규화 규칙 일치 확인  
- **CUDA 불일치**: 드라이버/툴킷/휠 버전을 일치시키거나 CPU 휠로 전환

---

## 🗺️ 로드맵(예시)

- [ ] **Late Fusion**: 카메라 예측 + 스마트워치 심박 후처리 결합  
- [ ] **로그 페어링**: 동일 타임스탬프의 얼굴/심박 로그 자동 페어링 → 데이터셋화  
- [ ] **Early/Hybrid Fusion**: 페어 데이터 축적 후 멀티모달 학습

---

## 🤝 기여

이슈/PR을 환영합니다. 다음 정보를 함께 남겨주세요.
- 실행한 커맨드(재현 가능 최소 단위)
- 환경 정보(OS, Python, GPU/CPU)
- 데이터 경로(또는 샘플/더미 데이터)
