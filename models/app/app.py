import json
import os
import streamlit as st
import wandb
from model_loader import load_production_model
from inference import run_inference

st.set_page_config(page_title="W&B Model Deployment Viewer", layout="wide")

st.title("W&B Model Deployment Viewer")

# --- WANDB_API_KEY 검증 ---
if not os.environ.get("WANDB_API_KEY"):
    st.error(
        "WANDB_API_KEY가 설정되지 않았습니다. "
        "Streamlit Cloud의 Secrets 설정에서 WANDB_API_KEY를 추가하세요."
    )
    st.code('WANDB_API_KEY = "your-wandb-api-key"', language="toml")
    st.stop()

# --- 배포 상태 로드 ---
DEPLOY_PATH = os.path.join(os.path.dirname(__file__), "deployment.json")

with open(DEPLOY_PATH) as f:
    deploy_info = json.load(f)

is_deployed = deploy_info.get("model_name", "none") != "none"

# --- 사이드바 ---
with st.sidebar:
    st.header("설정")
    model_type = st.selectbox("모델 유형", ["Classification"])
    st.markdown("---")
    if is_deployed:
        parts = deploy_info.get("artifact_path", "").split("/")
        if len(parts) >= 2:
            st.markdown(f"[W&B Project](https://wandb.ai/{parts[0]}/{parts[1]})")
        else:
            st.markdown("[W&B Dashboard](https://wandb.ai)")
    else:
        st.markdown("[W&B Dashboard](https://wandb.ai)")

# --- 배포 상태 ---
st.header("배포 상태")

if not is_deployed:
    st.warning("배포된 모델이 없습니다. W&B Model Registry에서 모델을 'production'으로 승격하세요.")
    st.stop()

col1, col2, col3 = st.columns(3)
col1.metric("모델", deploy_info["model_name"])
col2.metric("버전", deploy_info["model_version"])
col3.metric("배포 시각", deploy_info["deployed_at"])

col4, col5 = st.columns(2)
model_type_val = deploy_info.get("model_type", "-")
if model_type_val and model_type_val != "none":
    col4.metric("모델 유형", model_type_val)
best_acc = deploy_info.get("best_val_acc")
if best_acc is not None:
    col5.metric("Best Val Accuracy", f"{best_acc:.2%}")

# --- 모델 정보 (W&B API) ---
st.header("모델 정보")

try:
    api = wandb.Api()
    artifact = api.artifact(deploy_info["artifact_path"])

    # 메타데이터 표시
    col_meta1, col_meta2 = st.columns(2)
    with col_meta1:
        st.subheader("메타데이터")
        st.json(artifact.metadata)

    # 학습 run에서 메트릭 조회
    with col_meta2:
        st.subheader("학습 메트릭")
        run = artifact.logged_by()
        if run:
            metrics = {k: v for k, v in run.summary.items() if not k.startswith("_")}
            st.json(metrics)
        else:
            st.info("학습 run 정보를 찾을 수 없습니다.")
except Exception as e:
    st.error(f"W&B API 오류: {e}")
    st.info("WANDB_API_KEY 환경변수가 설정되어 있는지 확인하세요.")

# --- 추론 테스트 ---
st.header("추론 테스트")
uploaded = st.file_uploader("이미지를 업로드하세요 (어떤 크기든 가능)", type=["jpg", "png", "jpeg"])

if uploaded:
    try:
        model, metadata = load_production_model(deploy_info["artifact_path"])
        result = run_inference(model, uploaded, metadata)

        # 전처리 파이프라인 시각화
        st.subheader("전처리 파이프라인")
        h, w = result["input_size"]

        col_orig, col_arrow1, col_resized, col_arrow2, col_result = st.columns([3, 1, 2, 1, 3])

        with col_orig:
            orig = result["original_image"]
            st.image(orig, caption=f"원본 ({orig.width}x{orig.height})", use_container_width=True)

        with col_arrow1:
            st.markdown("<div style='text-align:center; padding-top:50%; font-size:2rem;'>→</div>", unsafe_allow_html=True)
            st.caption(f"Resize to {h}x{w}")

        with col_resized:
            st.image(result["resized_image"], caption=f"리사이즈 ({h}x{w})", use_container_width=True)

        with col_arrow2:
            st.markdown("<div style='text-align:center; padding-top:50%; font-size:2rem;'>→</div>", unsafe_allow_html=True)
            st.caption("Normalize")

        with col_result:
            st.metric("예측", result["top_class"])
            st.metric("신뢰도", f"{result['top_confidence']:.2%}")

        # Top-5 예측 결과
        st.subheader("Top-5 예측")
        for cls, prob in result["predictions"].items():
            prob_float = float(prob.strip("%")) / 100
            st.progress(prob_float, text=f"{cls}: {prob}")

    except Exception as e:
        st.error(f"추론 오류: {e}")

# --- 버전 이력 ---
st.header("버전 이력")

try:
    versions = artifact.collection.versions()
    history = []
    for v in versions:
        history.append({
            "버전": v.version,
            "Aliases": ", ".join(v.aliases) if v.aliases else "-",
            "등록 시각": str(v.created_at),
        })
    if history:
        st.dataframe(history, use_container_width=True)
    else:
        st.info("버전 이력이 없습니다.")
except Exception as e:
    st.warning(f"버전 이력을 불러올 수 없습니다: {e}")
