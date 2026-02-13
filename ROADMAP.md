# ROADMAP.md

W&B E2E Demo 구현 로드맵. 각 시나리오를 순서대로 구현한다.

---

## 1. Image Classification

> **경로**: `models/image_classification/image_classification.ipynb`

### 왜 첫 번째인가

가장 기본적인 CV 태스크로, W&B의 핵심 기능(Tracking, Artifacts, Sweep, Registry, Tables, Reports)을 자연스럽게 전부 보여줄 수 있다. 이후 시나리오들의 템플릿 역할을 한다.

### 데이터셋

- **CIFAR-10** (torchvision 내장, 별도 다운로드 불필요)
- 60,000장 (train 50k / test 10k), 32×32, 10 classes
- Colab에서 수 초 내 로드 완료

### 모델

- **ResNet-18** (torchvision pretrained → fine-tune)
- `torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)`
- 마지막 FC layer만 10-class로 교체

### 구현 상세

#### 셀 구성

| # | 셀 종류 | 내용 |
|---|---------|------|
| 1 | Code | `!pip install wandb torchvision` |
| 2 | Code | `wandb.login()` |
| 3 | Markdown | 프로젝트 소개 (한국어) |
| 4 | Code | Config 정의 (`batch_size`, `lr`, `epochs`, `optimizer` 등) |
| 5 | Code | 데이터 로드 + Transform 정의 + DataLoader 생성 |
| 6 | Code | **데이터셋 Artifact 생성** — `wandb.Artifact("cifar10", type="dataset")` 로 데이터셋 버저닝 |
| 7 | Code | 샘플 이미지를 `wandb.Table`로 시각화 (이미지 + 라벨 컬럼) |
| 8 | Code | 모델 정의 (ResNet-18, FC layer 수정) |
| 9 | Code | 학습 루프 — epoch마다 `wandb.log({"train/loss", "train/acc", "val/loss", "val/acc", "epoch"})` |
| 10 | Code | 검증 결과를 `wandb.Table`로 로깅 — 이미지, 정답, 예측, confidence |
| 11 | Code | **모델 Artifact 저장** — `wandb.Artifact("resnet18-cifar10", type="model")` + `artifact.add_file("model.pth")` |
| 12 | Code | **Model Registry 등록** — `run.link_artifact(artifact, "model-registry/cifar10-classifier", aliases=["staging"])` |
| 13 | Code | **Sweep 설정 및 실행** — lr, batch_size, optimizer를 sweep |
| 14 | Code | **Report 생성** — `wandb.apis.reports`로 실험 비교 리포트 자동 생성 |
| 15 | Code | `wandb.finish()` |

#### Sweep Config

```python
sweep_config = {
    "method": "bayes",
    "metric": {"name": "val/acc", "goal": "maximize"},
    "parameters": {
        "lr": {"min": 1e-5, "max": 1e-2, "distribution": "log_uniform_values"},
        "batch_size": {"values": [32, 64, 128]},
        "optimizer": {"values": ["adam", "sgd", "adamw"]},
    },
}
```

#### W&B 기능 커버리지

- [x] Experiment Tracking (config, metrics, system metrics)
- [x] Media Logging (wandb.Image)
- [x] Tables (데이터셋 미리보기, 예측 결과 비교)
- [x] Artifacts (데이터셋 + 모델 버저닝)
- [x] Model Registry (staging alias로 등록)
- [x] Sweeps (하이퍼파라미터 최적화)
- [x] Reports (프로그래밍 방식 리포트 생성)

---

## 2. Image Segmentation

> **경로**: `models/image_segmentation/image_segmentation.ipynb`

### 데이터셋

- **Oxford-IIIT Pet Dataset** (torchvision 내장)
- 약 7,400장, 37 breeds, **trimap segmentation mask 포함**
- 3-class segmentation: foreground / background / boundary
- Colab 친화적 크기

### 모델

- **DeepLabV3** with MobileNetV3-Large backbone (torchvision pretrained)
- `torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights=...)`
- classifier head만 3-class로 교체하여 fine-tune

### 구현 상세

#### 셀 구성

| # | 셀 종류 | 내용 |
|---|---------|------|
| 1 | Code | `!pip install wandb torchvision` |
| 2 | Code | `wandb.login()` |
| 3 | Markdown | Segmentation 태스크 소개 (한국어) |
| 4 | Code | Config 정의 (`lr`, `epochs`, `img_size=128`, `batch_size=16` 등) |
| 5 | Code | Oxford-IIIT Pet 데이터 로드 + Transform (Resize, Normalize, mask 전처리) |
| 6 | Code | **데이터셋 Artifact** — 샘플 이미지+마스크 쌍을 `wandb.Table`로 시각화 |
| 7 | Code | 모델 정의 (DeepLabV3, classifier 교체) |
| 8 | Code | 학습 루프 — `wandb.log({"train/loss", "val/loss", "val/mean_iou", "val/pixel_acc"})` |
| 9 | Code | **Segmentation 결과 시각화** — `wandb.Image`에 `masks` 파라미터로 예측 마스크 오버레이 로깅 |
| 10 | Code | 예측 결과 `wandb.Table` (원본, GT mask, Pred mask, IoU per image) |
| 11 | Code | **모델 Artifact + Registry 등록** |
| 12 | Code | Sweep (lr, backbone freeze 여부, img_size) |
| 13 | Code | Report 생성 |
| 14 | Code | `wandb.finish()` |

#### 핵심 시각화 — Segmentation Mask Overlay

```python
# W&B의 마스크 오버레이 기능 활용
wandb.log({"predictions": wandb.Image(
    image,
    masks={
        "predictions": {"mask_data": pred_mask, "class_labels": class_labels},
        "ground_truth": {"mask_data": gt_mask, "class_labels": class_labels},
    }
)})
```

#### Metrics

- **Mean IoU** (Intersection over Union) — 클래스별 + 평균
- **Pixel Accuracy** — 전체 픽셀 정확도
- **Dice Coefficient** — F1 at pixel level

#### W&B 기능 커버리지

- [x] Experiment Tracking
- [x] Media Logging (마스크 오버레이가 핵심 — W&B의 강점)
- [x] Tables (이미지-마스크 비교 테이블)
- [x] Artifacts (데이터셋 + 모델)
- [x] Model Registry
- [x] Sweeps
- [x] Reports

---

## 3. Image Detection

> **경로**: `models/image_detection/image_detection.ipynb`

### 데이터셋

- **PASCAL VOC 2012** (torchvision 내장)
- 약 11,500장, 20 object classes
- Bounding box + class label annotation 포함
- 또는 **COCO 128** (Ultralytics 제공 소형 subset) — 더 빠른 데모용

### 모델

- **YOLOv8** (Ultralytics) — `ultralytics` 패키지의 YOLO 클래스 사용
- pretrained `yolov8n.pt` (nano) 모델을 fine-tune
- Ultralytics는 W&B callback을 내장 지원

### 구현 상세

#### 셀 구성

| # | 셀 종류 | 내용 |
|---|---------|------|
| 1 | Code | `!pip install wandb ultralytics` |
| 2 | Code | `wandb.login()` |
| 3 | Markdown | Object Detection 소개 (한국어) |
| 4 | Code | 데이터셋 준비 (VOC 또는 COCO128 다운로드 + YOLO format 변환) |
| 5 | Code | **데이터셋 Artifact 생성** — 데이터셋 디렉토리를 Artifact로 버저닝 |
| 6 | Code | 샘플 데이터 `wandb.Table` 시각화 (이미지 + BBox 오버레이) |
| 7 | Code | **YOLOv8 학습** — `model.train(data=..., epochs=30, imgsz=640)` + W&B 자동 로깅 |
| 8 | Code | 학습 결과 수동 로깅 — `wandb.log({"val/mAP50", "val/mAP50-95", "val/precision", "val/recall"})` |
| 9 | Code | **Detection 결과 시각화** — `wandb.Image`에 `boxes` 파라미터로 BBox 오버레이 |
| 10 | Code | 예측 결과 `wandb.Table` (이미지, detected objects, confidence scores) |
| 11 | Code | **모델 Artifact + Registry 등록** |
| 12 | Code | Sweep (lr0, imgsz, augmentation 파라미터) |
| 13 | Code | Report 생성 |
| 14 | Code | `wandb.finish()` |

#### 핵심 시각화 — Bounding Box Overlay

```python
# W&B BBox 시각화
wandb.log({"detections": wandb.Image(
    image,
    boxes={
        "predictions": {
            "box_data": [
                {"position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2},
                 "class_id": cls, "scores": {"confidence": conf},
                 "box_caption": f"{class_name} {conf:.2f}"}
            ],
            "class_labels": class_labels,
        }
    }
)})
```

#### Metrics

- **mAP@0.5** — IoU 0.5 기준 mean Average Precision
- **mAP@0.5:0.95** — IoU 0.5~0.95 평균
- **Precision / Recall** — 클래스별 + 전체

#### W&B 기능 커버리지

- [x] Experiment Tracking (Ultralytics 자동 로깅 + 수동 로깅)
- [x] Media Logging (BBox 오버레이 — W&B의 강점)
- [x] Tables (Detection 결과 비교)
- [x] Artifacts
- [x] Model Registry
- [x] Sweeps
- [x] Reports

---

## 4. Video Classification

> **경로**: `models/video_classification/video_classification.ipynb`

### 데이터셋

- **UCF-101** 서브셋 (상위 10개 클래스만 사용, 약 1,300개 영상)
- 또는 **Kinetics-400** 미니 서브셋
- 영상을 16프레임으로 샘플링하여 사용

### 모델

- **VideoMAE** 또는 **R3D-18** (torchvision)
- `torchvision.models.video.r3d_18(weights=R3D_18_Weights.DEFAULT)` → FC 교체
- 또는 Hugging Face `VideoMAEForVideoClassification` pretrained → fine-tune

### 구현 상세

#### 셀 구성

| # | 셀 종류 | 내용 |
|---|---------|------|
| 1 | Code | `!pip install wandb torchvision pytorchvideo av` |
| 2 | Code | `wandb.login()` |
| 3 | Markdown | Video Classification 소개 (한국어) |
| 4 | Code | Config 정의 (`num_frames=16`, `clip_duration`, `lr`, `epochs` 등) |
| 5 | Code | UCF-101 서브셋 다운로드 + 프레임 샘플링 로직 |
| 6 | Code | **데이터셋 Artifact** — 비디오 샘플을 `wandb.Video`와 `wandb.Table`로 시각화 |
| 7 | Code | 모델 정의 (R3D-18, FC 교체) |
| 8 | Code | 학습 루프 — `wandb.log({"train/loss", "val/loss", "val/acc", "val/top5_acc"})` |
| 9 | Code | **비디오 예측 결과** — `wandb.Video`로 예측 결과 영상 로깅 + `wandb.Table` |
| 10 | Code | Confusion Matrix — `wandb.plot.confusion_matrix` |
| 11 | Code | **모델 Artifact + Registry 등록** |
| 12 | Code | Sweep (num_frames, lr, clip_duration) |
| 13 | Code | Report 생성 |
| 14 | Code | `wandb.finish()` |

#### 핵심 시각화 — Video Logging

```python
# W&B 비디오 로깅
wandb.log({
    "sample_videos": wandb.Video(video_array, fps=8, format="mp4"),
    "predictions_table": wandb.Table(
        columns=["Video", "Ground Truth", "Prediction", "Confidence"],
        data=[[wandb.Video(v), gt, pred, conf] for v, gt, pred, conf in results]
    )
})
```

#### Metrics

- **Top-1 Accuracy**
- **Top-5 Accuracy**
- **Per-class Accuracy**
- **Confusion Matrix**

#### W&B 기능 커버리지

- [x] Experiment Tracking
- [x] Media Logging (`wandb.Video` — 비디오 매체 로깅)
- [x] Tables (비디오 + 예측 결과)
- [x] Artifacts
- [x] Model Registry
- [x] Sweeps
- [x] Reports

---

## 5. LLM Fine-tuning (Qwen 3 4B)

> **경로**: `models/llm_finetuning/llm_finetuning.ipynb`

### 데이터셋

- **Korean Instruction Dataset** (예: `heegyu/korean-chatgpt-prompts` 또는 `beomi/KoAlpaca-v1.1a`)
- Hugging Face `datasets` 라이브러리로 로드
- Instruction-Response 포맷으로 전처리

### 모델

- **Qwen 3 4B** (`Qwen/Qwen3-4B`)
- **QLoRA** (4-bit quantization + LoRA) — Colab T4 GPU에서 실행 가능하도록
- `bitsandbytes` 4-bit, LoRA rank=16, alpha=32

### 구현 상세

#### 셀 구성

| # | 셀 종류 | 내용 |
|---|---------|------|
| 1 | Code | `!pip install wandb transformers peft bitsandbytes datasets trl accelerate` |
| 2 | Code | `wandb.login()` |
| 3 | Markdown | LLM Fine-tuning 소개 — QLoRA, PEFT 개념 설명 (한국어) |
| 4 | Code | Config 정의 (`lora_r=16`, `lora_alpha=32`, `lr=2e-4`, `epochs=3`, `max_seq_len=512` 등) |
| 5 | Code | 데이터셋 로드 + Instruction 템플릿 포맷팅 |
| 6 | Code | **데이터셋 Artifact** — 샘플 instruction-response 쌍을 `wandb.Table`로 시각화 |
| 7 | Code | 토크나이저 로드 + 데이터 토크나이징 |
| 8 | Code | 모델 로드 (4-bit quantization) + LoRA 적용 (`peft.get_peft_model`) |
| 9 | Code | 학습 가능 파라미터 수 확인 로깅 |
| 10 | Code | **SFTTrainer 학습** — HF Trainer + W&B integration (`report_to="wandb"`) |
| 11 | Code | 학습 중 metrics — `train/loss`, `eval/loss`, `eval/perplexity` 자동 로깅 |
| 12 | Code | **생성 결과 비교** — fine-tune 전/후 동일 프롬프트 응답을 `wandb.Table`로 비교 |
| 13 | Code | **모델 Artifact** — LoRA adapter weights 저장 + Artifact 등록 |
| 14 | Code | **Model Registry 등록** — `aliases=["staging"]` |
| 15 | Code | Sweep (lora_r, lr, epochs, max_seq_len) |
| 16 | Code | Report 생성 |
| 17 | Code | `wandb.finish()` |

#### QLoRA 설정

```python
from transformers import BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
```

#### Metrics

- **Training Loss** (step-level)
- **Eval Loss / Perplexity**
- **생성 품질** — 정성적 비교 (테이블)

#### W&B 기능 커버리지

- [x] Experiment Tracking (HF Trainer 자동 통합)
- [x] Tables (데이터셋 미리보기, 생성 결과 비교)
- [x] Artifacts (데이터셋 + LoRA adapter)
- [x] Model Registry
- [x] Sweeps (LoRA 하이퍼파라미터 탐색)
- [x] Reports

---

## 6. Image Generation (Text-to-Image)

> **경로**: `models/image_generation/image_generation.ipynb`

### 데이터셋

- **Hugging Face `lambdalabs/naruto-blip-captions`** (나루토 스타일 이미지 + 캡션, ~1,200장)
- 작은 크기로 Colab에서 빠르게 fine-tune 가능
- 또는 custom 이미지셋 + 캡션

### 모델

- **Stable Diffusion v1.5** (또는 SDXL-Turbo for speed)
- `diffusers` 라이브러리의 `StableDiffusionPipeline`
- **LoRA fine-tuning** — `diffusers` + `peft`로 UNet에 LoRA 적용

### 구현 상세

#### 셀 구성

| # | 셀 종류 | 내용 |
|---|---------|------|
| 1 | Code | `!pip install wandb diffusers transformers peft accelerate datasets` |
| 2 | Code | `wandb.login()` |
| 3 | Markdown | Text-to-Image 생성 모델 소개 (한국어) |
| 4 | Code | Config 정의 (`lr=1e-4`, `train_steps=500`, `resolution=512`, `lora_rank=4` 등) |
| 5 | Code | 데이터셋 로드 (naruto-blip-captions) + 전처리 |
| 6 | Code | **데이터셋 Artifact** — 샘플 이미지+캡션을 `wandb.Table`로 시각화 |
| 7 | Code | Stable Diffusion 파이프라인 로드 + LoRA 설정 |
| 8 | Code | **학습 루프** — step마다 `wandb.log({"train/loss"})` |
| 9 | Code | **생성 이미지 로깅** — N step마다 동일 프롬프트로 생성 → `wandb.Image`로 로깅 (생성 과정 추적) |
| 10 | Code | **최종 생성 결과** — 다양한 프롬프트로 이미지 생성 → `wandb.Table` (프롬프트, 생성 이미지, seed) |
| 11 | Code | **모델 Artifact** — LoRA weights 저장 + Artifact 등록 |
| 12 | Code | **Model Registry 등록** |
| 13 | Code | Sweep (lr, lora_rank, train_steps) |
| 14 | Code | Report 생성 |
| 15 | Code | `wandb.finish()` |

#### 핵심 시각화 — 생성 과정 추적

```python
# 학습 중 주기적으로 동일 프롬프트 생성 → 품질 변화 추적
eval_prompts = ["a naruto character with blue hair", "a village scene in anime style"]
if step % 100 == 0:
    images = pipeline(eval_prompts, num_inference_steps=25).images
    wandb.log({
        "generated": [wandb.Image(img, caption=p) for img, p in zip(images, eval_prompts)],
        "train/step": step,
    })
```

#### Metrics

- **Training Loss** (denoising loss)
- **생성 이미지 품질** — 정성적 (테이블로 프롬프트별 비교)
- **FID** (선택적, 시간이 허용되면)

#### W&B 기능 커버리지

- [x] Experiment Tracking
- [x] Media Logging (생성 이미지 추적 — generative AI에서 핵심)
- [x] Tables (프롬프트별 생성 결과 비교)
- [x] Artifacts
- [x] Model Registry
- [x] Sweeps
- [x] Reports

---

## 7-8. Automations + Streamlit 배포 (GitHub Actions 연동)

> 7(Automations)과 8(Streamlit App)은 하나의 파이프라인으로 함께 구현한다.

### 전체 아키텍처

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Notebook   │     │  W&B Model       │     │  GitHub Actions     │     │  Streamlit       │
│  (Colab)    │────▶│  Registry        │────▶│  Workflow           │────▶│  Cloud           │
│             │     │                  │     │                     │     │                  │
│ 학습 완료   │     │ "production"     │     │ 모델 다운로드       │     │ 새 모델로 서빙   │
│ → 모델 등록 │     │ alias 승격 시    │     │ → 앱 재배포         │     │ → 추론 테스트    │
│ → 승격      │     │ Webhook 발동     │     │                     │     │                  │
└─────────────┘     └──────────────────┘     └─────────────────────┘     └──────────────────┘
```

### 파일 구조

```
models/
├── automations/
│   └── automations.ipynb                  # 모델 승격 + Automation 설정 가이드
├── app/
│   ├── app.py                             # Streamlit 메인 앱
│   ├── model_loader.py                    # W&B Artifact → 모델 로드 유틸
│   ├── inference.py                       # 유스케이스별 추론 로직
│   └── requirements.txt                   # streamlit, wandb, torch, torchvision, ultralytics, Pillow
└── .github/
    └── workflows/
        └── deploy-on-promotion.yml        # GitHub Actions 워크플로우
```

---

### Part A: GitHub Actions 워크플로우

> **경로**: `.github/workflows/deploy-on-promotion.yml`

W&B Automation의 Webhook이 GitHub `repository_dispatch` 이벤트를 트리거하면 실행된다.

#### 워크플로우 상세

```yaml
name: Deploy Model on Production Promotion

on:
  repository_dispatch:
    types: [wandb-model-promoted]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      # 1. Checkout
      - uses: actions/checkout@v4

      # 2. Python 설정
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      # 3. 의존성 설치
      - run: pip install wandb streamlit

      # 4. W&B에서 모델 정보 추출
      - name: Parse webhook payload
        run: |
          echo "MODEL_NAME=${{ github.event.client_payload.model_name }}" >> $GITHUB_ENV
          echo "MODEL_VERSION=${{ github.event.client_payload.model_version }}" >> $GITHUB_ENV
          echo "ARTIFACT_PATH=${{ github.event.client_payload.artifact_path }}" >> $GITHUB_ENV

      # 5. 모델 Artifact 다운로드 + 검증
      - name: Download and validate model
        env:
          WANDB_API_KEY: ${{ secrets.WANDB_API_KEY }}
        run: |
          python -c "
          import wandb
          api = wandb.Api()
          artifact = api.artifact('${{ env.ARTIFACT_PATH }}')
          artifact.download('./deployed_model')
          print(f'Downloaded: {artifact.name} v{artifact.version}')
          "

      # 6. Streamlit Cloud 재배포 트리거
      #    (Streamlit Cloud는 GitHub repo push 시 자동 재배포)
      - name: Update deployment manifest
        run: |
          echo '{
            "model_name": "${{ env.MODEL_NAME }}",
            "model_version": "${{ env.MODEL_VERSION }}",
            "artifact_path": "${{ env.ARTIFACT_PATH }}",
            "deployed_at": "'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"
          }' > models/app/deployment.json

      - name: Commit and push deployment update
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add models/app/deployment.json
          git commit -m "deploy: ${{ env.MODEL_NAME }} ${{ env.MODEL_VERSION }}"
          git push
```

#### W&B Automation → GitHub Webhook 연결 방법

W&B UI에서 설정:

1. **Settings → Webhooks** 에서 새 Webhook 생성
   - URL: `https://api.github.com/repos/{owner}/{repo}/dispatches`
   - Auth: `Bearer {GITHUB_PAT}` (repo scope 필요)
   - Payload template:
     ```json
     {
       "event_type": "wandb-model-promoted",
       "client_payload": {
         "model_name": "${event.artifact_collection_name}",
         "model_version": "${event.artifact_version}",
         "artifact_path": "${event.artifact_version_string}",
         "event_author": "${event.event_author}"
       }
     }
     ```

2. **Automations → New Automation** 에서:
   - 이벤트: **An alias is added to an artifact version in a registered model**
   - 필터: alias = `production`
   - 액션: 위에서 만든 Webhook 선택

---

### Part B: Automations 노트북

> **경로**: `models/automations/automations.ipynb`

Colab에서 실행. Model Registry 승격을 트리거하고, 파이프라인 동작을 확인한다.

#### 셀 구성

| # | 셀 종류 | 내용 |
|---|---------|------|
| 1 | Code | `!pip install wandb` |
| 2 | Code | `wandb.login()` |
| 3 | Markdown | **파이프라인 아키텍처 설명** (한국어) — 전체 흐름도, 각 컴포넌트 역할 |
| 4 | Markdown | **사전 준비 사항** — GitHub PAT 발급, W&B Webhook 설정, Streamlit Cloud 연동 가이드 |
| 5 | Code | **Registry 현황 조회** — `wandb.Api()`로 등록된 모델 목록 + 현재 alias 확인 |
| 6 | Code | **모델 성능 비교** — staging 모델들의 메트릭을 `wandb.Table`로 비교하여 승격 대상 선정 |
| 7 | Code | **"production" 승격 실행** — 선택한 모델 버전에 "production" alias 추가 |
| 8 | Code | **Automation 트리거 확인** — GitHub Actions API로 워크플로우 실행 상태 확인 |
| 9 | Code | **배포 상태 확인** — Streamlit 앱 URL 헬스체크 + deployment.json 내용 확인 |
| 10 | Code | **배포 이력 로깅** — 배포 이벤트를 wandb run으로 기록 |
| 11 | Markdown | **롤백 가이드** — 이전 버전을 "production"으로 재승격하는 방법 |
| 12 | Code | **롤백 실행 예시** — 이전 버전 Artifact에 "production" alias 이동 |

#### 핵심 코드 — 모델 승격

```python
import wandb
api = wandb.Api()

# Registry에 등록된 모델 조회
collections = api.artifact_type("model", project="wandb-e2e-demo-image-classification").collections()
for c in collections:
    print(f"Model: {c.name}")
    for v in c.versions():
        print(f"  {v.version} aliases={v.aliases}")

# staging → production 승격
artifact = api.artifact("my-entity/wandb-e2e-demo-image-classification/cifar10-classifier:v3")
artifact.aliases.append("production")
artifact.save()
print(f"✅ {artifact.name}:{artifact.version} → production 승격 완료")
# 이 시점에서 W&B Automation이 GitHub Actions Webhook을 트리거
```

#### 핵심 코드 — GitHub Actions 상태 확인

```python
import requests

GITHUB_TOKEN = "..."  # userdata.get('GITHUB_PAT')
REPO = "owner/wandb-e2e-demo"

resp = requests.get(
    f"https://api.github.com/repos/{REPO}/actions/runs",
    headers={"Authorization": f"Bearer {GITHUB_TOKEN}"},
    params={"event": "repository_dispatch", "per_page": 3}
)
for run in resp.json()["workflow_runs"]:
    print(f"  [{run['status']}] {run['name']} - {run['created_at']}")
```

---

### Part C: Streamlit 배포 뷰어 앱

> **경로**: `models/app/`

GitHub Actions가 `deployment.json`을 업데이트 → Streamlit Cloud가 자동 재배포 → 앱이 최신 production 모델로 서빙.

#### 앱 파일 구조

```
models/app/
├── app.py                 # 메인 Streamlit 앱
├── model_loader.py        # W&B Artifact 다운로드 + 모델 인스턴스 로드
├── inference.py           # 유스케이스별 추론 함수 (classify, detect, segment)
├── deployment.json        # GitHub Actions가 업데이트하는 배포 매니페스트
└── requirements.txt
```

#### app.py 페이지 구성

| 페이지/섹션 | 내용 |
|-------------|------|
| **사이드바** | 모델 유형 선택 (Classification / Detection / Segmentation), W&B 프로젝트 링크 |
| **배포 상태** | `deployment.json`에서 현재 배포된 모델 이름, 버전, 배포 시각 표시 |
| **모델 정보** | W&B API로 해당 모델의 메트릭 (accuracy, mAP 등), 학습 config, lineage 표시 |
| **추론 테스트** | 이미지/비디오 업로드 → 모델 추론 → 결과 시각화 (클래스, BBox, 마스크 등) |
| **버전 이력** | Model Registry의 버전 히스토리 테이블 (버전, alias, 등록자, 등록 시각, 주요 메트릭) |

#### app.py 핵심 구조

```python
import streamlit as st
import wandb
import json
from model_loader import load_production_model
from inference import run_inference

st.set_page_config(page_title="W&B Model Deployment Viewer", layout="wide")

# --- 배포 상태 ---
with open("deployment.json") as f:
    deploy_info = json.load(f)

st.header("🚀 배포 상태")
col1, col2, col3 = st.columns(3)
col1.metric("모델", deploy_info["model_name"])
col2.metric("버전", deploy_info["model_version"])
col3.metric("배포 시각", deploy_info["deployed_at"])

# --- 모델 정보 (W&B API) ---
api = wandb.Api()
artifact = api.artifact(deploy_info["artifact_path"])

st.subheader("📊 모델 메트릭")
# artifact.logged_by() 로 원본 run에서 메트릭 조회
run = artifact.logged_by()
if run:
    st.json({k: v for k, v in run.summary.items() if not k.startswith("_")})

# --- 추론 테스트 ---
st.header("🔍 추론 테스트")
uploaded = st.file_uploader("이미지를 업로드하세요", type=["jpg", "png", "jpeg"])
if uploaded:
    model = load_production_model(deploy_info["artifact_path"])
    result = run_inference(model, uploaded, model_type=deploy_info["model_name"])

    col_in, col_out = st.columns(2)
    with col_in:
        st.image(uploaded, caption="입력 이미지")
    with col_out:
        st.image(result["visualization"], caption="추론 결과")
        st.write(result["details"])

# --- 버전 이력 ---
st.header("📋 버전 이력")
collection = api.artifact_collection("model", f"{artifact.entity}/{artifact.project}/{deploy_info['model_name']}")
history = []
for v in collection.versions():
    history.append({
        "버전": v.version,
        "Aliases": ", ".join(v.aliases),
        "등록 시각": str(v.created_at),
    })
st.dataframe(history, use_container_width=True)
```

#### model_loader.py

```python
import wandb
import torch
import streamlit as st

@st.cache_resource
def load_production_model(artifact_path: str):
    """W&B Artifact에서 production 모델을 다운로드하고 로드"""
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()

    # 모델 유형에 따라 분기
    metadata = artifact.metadata
    model_type = metadata.get("model_type", "classification")

    if model_type == "classification":
        from torchvision.models import resnet18
        model = resnet18(num_classes=metadata.get("num_classes", 10))
        model.load_state_dict(torch.load(f"{artifact_dir}/model.pth", map_location="cpu"))
    elif model_type == "detection":
        from ultralytics import YOLO
        model = YOLO(f"{artifact_dir}/best.pt")
    elif model_type == "segmentation":
        from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large
        model = deeplabv3_mobilenet_v3_large(num_classes=metadata.get("num_classes", 3))
        model.load_state_dict(torch.load(f"{artifact_dir}/model.pth", map_location="cpu"))

    return model
```

#### inference.py

```python
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def run_inference(model, uploaded_file, model_type: str) -> dict:
    image = Image.open(uploaded_file).convert("RGB")

    if model_type == "classification":
        return _classify(model, image)
    elif model_type == "detection":
        return _detect(model, image)
    elif model_type == "segmentation":
        return _segment(model, image)

def _classify(model, image):
    transform = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    model.eval()
    with torch.no_grad():
        output = model(transform(image).unsqueeze(0))
        probs = torch.softmax(output, dim=1)[0]
        top5 = torch.topk(probs, 5)
    return {
        "visualization": image,
        "details": {f"Class {i.item()}": f"{p.item():.2%}" for p, i in zip(top5.values, top5.indices)}
    }

def _detect(model, image):
    results = model(image)
    plotted = results[0].plot()  # Ultralytics 내장 시각화
    return {"visualization": plotted, "details": f"{len(results[0].boxes)}개 객체 검출"}

def _segment(model, image):
    transform = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    model.eval()
    with torch.no_grad():
        output = model(transform(image).unsqueeze(0))["out"]
        mask = output.argmax(1).squeeze().numpy()
    # 마스크 컬러 오버레이
    overlay = np.array(image.resize((224,224)))
    overlay[mask == 1] = overlay[mask == 1] * 0.5 + np.array([0,255,0]) * 0.5
    return {"visualization": Image.fromarray(overlay.astype(np.uint8)), "details": "Segmentation 완료"}
```

#### requirements.txt

```
streamlit>=1.30.0
wandb>=0.16.0
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
Pillow>=10.0.0
```

---

### 데모 시나리오 흐름 (라이브 데모용)

```
① [Colab] 1~6번 노트북 중 하나로 학습 완료, 모델이 Registry에 "staging"으로 등록됨
      ↓
② [W&B UI] Model Registry에서 모델 메트릭 확인 (청중과 함께 리뷰)
      ↓
③ [Colab] automations.ipynb에서 해당 모델을 "production"으로 승격
      ↓
④ [W&B UI] Automations 탭에서 Webhook 발동 로그 확인
      ↓
⑤ [GitHub] Actions 탭에서 deploy-on-promotion 워크플로우 실행 확인
      ↓
⑥ [Streamlit] 앱이 자동 재배포 → 새 모델 정보 표시 → 이미지 업로드하여 추론 테스트
      ↓
⑦ (선택) 롤백 시연 — 이전 버전을 다시 "production"으로 → 파이프라인 재실행
```

---

## 구현 순서 요약

```
1. Image Classification  ← 기본 템플릿 수립, W&B 전 기능 커버
2. Image Segmentation    ← 마스크 오버레이 시각화
3. Image Detection       ← BBox 시각화, YOLOv8 연동
4. Video Classification  ← 비디오 매체 로깅
5. LLM Fine-tuning       ← QLoRA, HF Trainer 통합
6. Image Generation      ← 생성 모델, 이미지 품질 추적
7-8. Automations + Streamlit App ← Registry→GitHub Actions→Streamlit Cloud 자동 배포 파이프라인
```

1~6은 독립 실행 가능. 7-8은 1~6에서 등록한 모델을 활용하며, 하나의 통합 파이프라인으로 구현한다.
