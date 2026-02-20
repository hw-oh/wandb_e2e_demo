# W&B E2E Demo

W&B(Weights & Biases) 플랫폼의 전체 기능을 시연하는 End-to-End 데모 레포지토리입니다.
모든 데모는 Google Colab에서 바로 실행 가능한 Jupyter 노트북으로 구성되어 있습니다.

---

## 데모 범위

하나의 노트북에서 **학습 → 실험 추적 → 모델 등록 → 자동 배포**까지 풀 플로우를 커버합니다.

```
학습 & 추적 → Artifacts → Model Registry → Sweep → Report → Automations → 배포
```

### 다루는 W&B 기능

| 기능 | 설명 |
|------|------|
| **Experiment Tracking** | `wandb.init`, `wandb.log`, `wandb.config`로 메트릭/시스템 메트릭 실시간 추적 |
| **Media Logging** | `wandb.Image`, `wandb.Video`로 예측 결과 시각화 |
| **Tables** | `wandb.Table`로 데이터셋 미리보기, 예측 결과 비교 (필터링/정렬) |
| **Artifacts** | 데이터셋/모델 버저닝, Lineage(계보) 추적 |
| **Model Registry** | 모델 등록, alias 관리 (staging → production) |
| **Sweeps** | Bayesian 하이퍼파라미터 최적화 |
| **Reports** | 프로그래밍 방식 실험 리포트 생성 |
| **Automations** | production 승격 → Webhook → GitHub Actions → Streamlit 자동 배포 |

---

## 유스케이스

| # | 유스케이스 | 모델 | 데이터셋 | 핵심 시각화 |
|---|-----------|------|----------|------------|
| 1 | Image Classification | ResNet-18 | CIFAR-10 | 클래스별 softmax 확률 테이블 |
| 2 | Image Segmentation | DeepLabV3 | Oxford-IIIT Pet | 마스크 오버레이 (`wandb.Image` masks) |
| 3 | Image Detection | YOLOv8 | VOC/COCO128 | BBox 오버레이 (`wandb.Image` boxes) |
| 4 | Video Classification | R3D-18 | UCF-101 subset | `wandb.Video` 비디오 로깅 |
| 5 | LLM Fine-tuning | Qwen 3 4B (QLoRA) | Korean Instruction | 생성 텍스트 전/후 비교 테이블 |
| 6 | Image Generation | Stable Diffusion (LoRA) | naruto-blip-captions | 생성 과정 이미지 추적 |

---

## 레포지토리 구조

```
wandb_e2e_demo/
├── models/
│   ├── image_classification/
│   │   └── image_classification.ipynb    ← 현재 구현 완료
│   ├── image_segmentation/               ← 예정
│   ├── image_detection/                  ← 예정
│   ├── video_classification/             ← 예정
│   ├── llm_finetuning/                   ← 예정
│   ├── image_generation/                 ← 예정
│   ├── automations/
│   │   └── automations.ipynb             ← 모델 비교/승격 허브 (선택)
│   └── app/
│       ├── app.py                        ← Streamlit 배포 뷰어 앱
│       ├── model_loader.py
│       ├── inference.py
│       └── requirements.txt
├── weave/                                ← W&B Weave 데모 (예정)
├── .github/workflows/
│   └── deploy-on-promotion.yml           ← GitHub Actions 자동 배포
├── SETUP.md                              ← 환경 설정 가이드
├── ROADMAP.md                            ← 구현 로드맵
└── CHECKLIST.md                          ← 데모 체크리스트
```

---

## 자동 배포 파이프라인

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Colab       │     │  W&B Model       │     │  GitHub Actions     │     │  Streamlit       │
│  노트북      │───▶│  Registry        │───▶│  Workflow           │───▶│  Cloud           │
│             │     │                  │     │                     │     │                  │
│ "production" │     │  Webhook 발동    │     │  deployment.json    │     │  자동 재배포     │
│  승격        │     │                  │     │  업데이트           │     │  → 추론 가능     │
└─────────────┘     └──────────────────┘     └─────────────────────┘     └──────────────────┘
```

---

## 빠른 시작

### 1. 환경 설정

[SETUP.md](SETUP.md)를 따라 Colab Secrets, W&B Webhook, GitHub Actions, Streamlit Cloud를 설정합니다.

### 2. 노트북 실행

Colab에서 노트북을 열고 위에서 아래로 순서대로 실행합니다.

- **파일 → GitHub에서 노트북 열기** → 이 레포를 선택 → 노트북 선택
- 런타임 → 런타임 유형 변경 → **GPU (T4)** 선택

### 3. 데모 플로우

```
① 노트북 실행 → 학습 + 실험 추적 + Sweep + Report
② W&B 대시보드에서 결과 확인
③ 노트북 하단 Automations 섹션에서 "production" 승격
④ GitHub Actions 자동 실행 → Streamlit 앱 재배포
⑤ Streamlit 앱에서 추론 테스트
```

---

## 기술 스택

- **Python 3.10+** / **PyTorch**
- **Hugging Face Transformers / PEFT** — LLM fine-tuning
- **Diffusers** — Image generation
- **Ultralytics** — Object detection
- **W&B SDK** (`wandb`, `wandb-workspaces`)
- **Streamlit** — 배포 뷰어 앱
- **Google Colab** — 실행 환경
- **GitHub Actions** — CI/CD 파이프라인

---

## 관련 문서

| 문서 | 설명 |
|------|------|
| [SETUP.md](SETUP.md) | Colab Secrets, W&B Webhook, GitHub Actions, Streamlit Cloud 설정 가이드 |
| [ROADMAP.md](ROADMAP.md) | 각 유스케이스의 상세 구현 스펙 |
| [CHECKLIST.md](CHECKLIST.md) | 데모에서 W&B 기능을 빠짐없이 보여주었는지 확인하는 체크리스트 |
