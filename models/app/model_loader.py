import streamlit as st
import wandb
from ultralytics import YOLO


@st.cache_resource
def load_production_model(artifact_path: str):
    """W&B Artifact에서 production YOLO 모델을 다운로드하고 로드한다."""
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()
    metadata = artifact.metadata
    model = YOLO(f"{artifact_dir}/best.pt")
    return model, metadata
