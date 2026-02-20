import json
import os
from urllib.parse import quote
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
    st.markdown("**YOLOv8-seg 통합 모델**")
    st.caption("한 번의 추론으로 Classification, Detection, Segmentation 결과를 모두 제공합니다.")
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

col4, col5, col6 = st.columns(3)
model_type_val = deploy_info.get("model_type", "-")
if model_type_val and model_type_val != "none":
    col4.metric("모델 유형", model_type_val)
best_map50 = deploy_info.get("best_mAP50")
best_map50_mask = deploy_info.get("best_mAP50_mask")
if best_map50 is not None:
    col5.metric("mAP@0.5 (Box)", f"{best_map50:.4f}")
if best_map50_mask is not None:
    col6.metric("mAP@0.5 (Mask)", f"{best_map50_mask:.4f}")

# Registry 링크
artifact_path = deploy_info.get("artifact_path", "")
if artifact_path and artifact_path != "none":
    parts = artifact_path.split("/")
    if len(parts) == 3 and ":" in parts[2]:
        entity = parts[0]
        project = parts[1]
        collection, version = parts[2].rsplit(":", 1)
        registry_name = project.removeprefix("wandb-registry-")
        org = os.environ.get("WANDB_ORG", entity)
        selection_path = f"{entity}/{project}/{collection}"
        registry_url = (
            f"https://wandb.ai/orgs/{quote(org)}/registry/{quote(registry_name)}"
            f"?selectionPath={quote(selection_path, safe='')}"
            f"&view=membership&tab=overview&version={version}"
        )
        st.markdown(f"[W&B Registry에서 보기 — {collection}:{version}]({registry_url})")
    else:
        st.code(artifact_path, language=None)

# --- 추론 테스트 ---
st.header("추론 테스트")
uploaded = st.file_uploader("이미지를 업로드하세요 (어떤 크기든 가능)", type=["jpg", "png", "jpeg"])

if uploaded:
    try:
        model, metadata = load_production_model(deploy_info["artifact_path"])
        result = run_inference(model, uploaded, metadata)

        tab_cls, tab_det, tab_seg = st.tabs(["Classification", "Detection", "Segmentation"])

        with tab_cls:
            st.subheader("검출된 클래스")
            if result["class_summary"]:
                for cls_name, info in result["class_summary"].items():
                    col_a, col_b, col_c = st.columns([3, 1, 1])
                    col_a.write(f"**{cls_name}**")
                    col_b.write(f"{info['count']}개")
                    col_c.write(f"최고 {info['max_conf']:.1%}")
            else:
                st.info("검출된 객체가 없습니다.")

        with tab_det:
            st.subheader(f"객체 탐지 결과 ({result['num_objects']}개)")
            col_orig, col_det = st.columns(2)
            with col_orig:
                st.image(result["original_image"], caption="원본", use_container_width=True)
            with col_det:
                st.image(result["det_image"], caption="탐지 결과", use_container_width=True)

            if result["detections"]:
                st.dataframe(result["detections"], use_container_width=True)

        with tab_seg:
            st.subheader("인스턴스 세그멘테이션 결과")
            if result["seg_image"] is not None:
                col_orig2, col_seg = st.columns(2)
                with col_orig2:
                    st.image(result["original_image"], caption="원본", use_container_width=True)
                with col_seg:
                    st.image(result["seg_image"], caption="세그멘테이션 마스크", use_container_width=True)
            else:
                st.info("이 이미지에서 세그멘테이션 마스크가 생성되지 않았습니다.")

    except Exception as e:
        st.error(f"추론 오류: {e}")

# --- 버전 이력 ---
st.header("버전 이력")

try:
    api = wandb.Api()
    artifact = api.artifact(deploy_info["artifact_path"])
    collection_path = deploy_info["artifact_path"].rsplit(":", 1)[0]
    versions = api.artifacts(type_name=artifact.type, name=collection_path)
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
