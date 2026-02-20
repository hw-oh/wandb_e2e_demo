import torch
import torch.nn as nn
import streamlit as st
import wandb
from torchvision.models import resnet18
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large


def _create_resnet18_cifar10(num_classes=10):
    """CIFAR-10(32x32) 입력에 맞게 수정한 ResNet-18.

    반드시 학습 시와 동일한 아키텍처여야 state_dict 로드가 가능하다.
    - conv1: 3x3, stride=1, padding=1 (표준: 7x7, stride=2)
    - maxpool: Identity (표준: 3x3, stride=2)
    - fc: num_classes (표준: 1000)
    """
    model = resnet18()
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _create_deeplabv3_mobilenetv3(num_classes=3):
    """DeepLabV3 + MobileNetV3-Large, classifier head를 num_classes로 교체.

    weights=None으로 빈 모델을 만든 뒤 state_dict를 로드해야 한다.
    학습 노트북과 동일한 아키텍처를 재현한다.
    """
    model = deeplabv3_mobilenet_v3_large(weights=None, num_classes=21)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[-1] = nn.Conv2d(10, num_classes, kernel_size=1)
    return model


@st.cache_resource
def load_production_model(artifact_path: str):
    """W&B Artifact에서 production 모델을 다운로드하고 로드한다."""
    api = wandb.Api()
    artifact = api.artifact(artifact_path)
    artifact_dir = artifact.download()

    metadata = artifact.metadata
    model_type = metadata.get("model_type", "classification")

    if model_type == "classification":
        num_classes = metadata.get("num_classes", 10)
        model = _create_resnet18_cifar10(num_classes)
        model.load_state_dict(
            torch.load(f"{artifact_dir}/model.pth", map_location="cpu", weights_only=True)
        )
        model.eval()
    elif model_type == "segmentation":
        num_classes = metadata.get("num_classes", 3)
        model = _create_deeplabv3_mobilenetv3(num_classes)
        model.load_state_dict(
            torch.load(f"{artifact_dir}/model.pth", map_location="cpu", weights_only=True)
        )
        model.eval()
    elif model_type == "detection":
        from ultralytics import YOLO
        model = YOLO(f"{artifact_dir}/best.pt")
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not yet supported.")

    return model, metadata
