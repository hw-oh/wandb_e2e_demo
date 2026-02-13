# Setup Guide — W&B E2E Demo

W&B Model Registry → GitHub Actions → Streamlit 자동 배포 파이프라인을 구성하는 가이드입니다.

---

## Prerequisites

- [W&B](https://wandb.ai) 계정
- [GitHub](https://github.com) 계정
- [Google Colab](https://colab.research.google.com) 접근
- [Streamlit Cloud](https://share.streamlit.io) 계정 (무료)

---

## Step 1: Google Colab Secrets 설정

Colab 노트북 좌측 **🔑 아이콘 → Secrets** 에서 아래 키들을 등록합니다.

| Secret 이름 | 설명 | 예시 |
|---|---|---|
| `WANDB_API_KEY` | W&B API 키 ([wandb.ai/authorize](https://wandb.ai/authorize)) | `abc123...` |
| `WANDB_ENTITY` | W&B 사용자명 또는 팀명 | `my-team` |
| `WANDB_PROJECT` | W&B 프로젝트명 | `wandb-e2e-demo-image-classification` |
| `GITHUB_PAT` | GitHub Personal Access Token (`repo` scope) | `ghp_xxx...` |
| `GITHUB_REPO` | GitHub 레포 (owner/repo 형식) | `username/wandb_e2e_demo` |

### GitHub PAT 생성 방법

1. GitHub → Settings → Developer settings → Personal access tokens → **Tokens (classic)**
2. **Generate new token (classic)** 클릭
3. Scope: `repo` 체크
4. 생성된 토큰을 Colab Secrets에 `GITHUB_PAT`로 저장

---

## Step 2: GitHub Repository Secrets 설정

GitHub 레포 → **Settings → Secrets and variables → Actions → New repository secret**

| Secret 이름 | 설명 |
|---|---|
| `WANDB_API_KEY` | W&B API 키 (Step 1과 동일한 값) |

이 시크릿은 GitHub Actions 워크플로우에서 W&B Artifact를 검증하고 배포 이벤트를 기록하는 데 사용됩니다.

---

## Step 3: W&B Webhook 설정

W&B → **Team Settings → Webhooks → New Webhook**

| 항목 | 값 |
|---|---|
| **Name** | `GitHub Actions Deploy` |
| **URL** | `https://api.github.com/repos/{owner}/{repo}/dispatches` |
| **Secret** | _(비워두기)_ |
| **Access Token** | GitHub PAT (Step 1에서 생성한 토큰) |

### Payload Template

```json
{
  "event_type": "wandb-model-promoted",
  "client_payload": {
    "event_author": "${event_author}",
    "artifact_version_string": "${artifact_version_string}",
    "artifact_collection_name": "${artifact_collection_name}",
    "entity_name": "${entity_name}",
    "project_name": "${project_name}"
  }
}
```

### 사용 가능한 템플릿 변수

| 변수 | 런타임 값 예시 | 설명 |
|---|---|---|
| `${artifact_collection_name}` | `cifar10-classifier` | Registry에 등록된 모델 이름 |
| `${artifact_version_string}` | `hw-oh/model-registry/cifar10-classifier:production` | Artifact 경로 (문자열) |
| `${artifact_version_index}` | `3` | 버전 인덱스 (정수) |
| `${event_author}` | `hw-oh` | 승격을 실행한 사용자 |
| `${entity_name}` | `hw-oh` | Entity 이름 |
| `${project_name}` | `model-registry` | **주의:** Model Registry 프로젝트명 (사용자 프로젝트가 아님) |
| `${alias}` | `production` | 추가된 alias |
| `${artifact_metadata.KEY}` | _(메타데이터 값)_ | Artifact 메타데이터 (top-level만) |
| `${artifact_version}` | `wandb-artifact://_id/QXJ0aWZ...` | Artifact 내부 참조 ID |

> **참고:** URL의 `{owner}/{repo}`를 실제 레포 경로로 교체하세요 (예: `https://api.github.com/repos/hw-oh/wandb_e2e_demo/dispatches`).

---

## Step 4: W&B Automation Rule 설정

W&B → **Automations → New Automation**

| 항목 | 값 |
|---|---|
| **Name** | `Deploy on Production Promotion` |
| **Event** | An alias is added to an artifact version in a registered model |
| **Filter** | Alias = `production` |
| **Action** | Webhooks → `GitHub Actions Deploy` (Step 3에서 생성한 Webhook) |

이 규칙이 활성화되면, Model Registry에서 어떤 모델 버전에 `production` alias를 추가할 때마다 자동으로 GitHub Actions 배포 파이프라인이 트리거됩니다.

---

## Step 5: Streamlit Cloud 설정

### 앱 배포

1. [share.streamlit.io](https://share.streamlit.io) 접속 → **New app**
2. GitHub 레포 연결: `{owner}/{repo}`
3. **Branch**: `main`
4. **Main file path**: `models/app/app.py`
5. **Deploy** 클릭

### Secrets 설정

Streamlit Cloud → App → **Settings → Secrets**

```toml
WANDB_API_KEY = "your-wandb-api-key"
WANDB_ORG = "your-wandb-org-name"
```

| Secret | 설명 | 예시 |
|---|---|---|
| `WANDB_API_KEY` | W&B API 키 | `abc123...` |
| `WANDB_ORG` | W&B Organization 이름 (Registry 링크용) | `wandb` |

Streamlit Cloud는 Secrets를 환경변수로 자동 주입합니다.

---

## 전체 흐름

```
1. Colab에서 image_classification.ipynb 실행 → 모델 학습 + Registry에 "staging" 등록
2. W&B UI → Model Registry → 모델 버전에 "production" alias 추가
3. W&B Automation이 Webhook 발동 → GitHub Actions repository_dispatch
4. GitHub Actions:
   - Artifact 메타데이터 검증
   - deployment.json 업데이트
   - commit & push
   - 배포 이벤트를 W&B에 기록
5. Streamlit Cloud가 git push 감지 → 자동 재배포
6. Streamlit 앱에서 새 모델로 추론 가능
```

---

## 검증

### 수동 트리거 (GitHub CLI)

Webhook 없이 GitHub Actions 워크플로우를 직접 테스트할 수 있습니다:

```bash
gh api repos/{owner}/{repo}/dispatches \
  -f event_type=wandb-model-promoted \
  -f 'client_payload[model_name]=cifar10-classifier' \
  -f 'client_payload[model_version]=v0' \
  -f 'client_payload[artifact_path]=entity/project/resnet18-cifar10:v0' \
  -f 'client_payload[event_author]=manual-test'
```

### Automations 노트북

`models/automations/automations.ipynb`에서 프로그래밍적으로 모델 승격 및 배포 상태 확인이 가능합니다.
