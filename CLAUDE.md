# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

W&B (Weights & Biases) end-to-end demo repository for pre-sales engineers. All demos are Jupyter notebooks designed to run on Google Colab. The project showcases W&B's full platform capabilities across two main product areas: **Models** and **Weave**.

## Repository Structure

```
wandb_e2e_demo/
├── models/          # W&B Models demos (training, tracking, registry, deployment)
│   ├── image_segmentation/
│   ├── image_classification/
│   ├── image_detection/
│   ├── video_classification/
│   ├── llm_finetuning/        # Qwen 3 4B fine-tuning
│   ├── image_generation/      # Text-to-image
│   ├── automations/           # Auto-deploy via W&B Automations
│   └── app/                   # Streamlit deployment viewer app
├── weave/           # W&B Weave demos (LLM observability & evaluation)
└── CLAUDE.md
```

## Models Demos — Use Cases

Each use case is a standalone Jupyter notebook (`.ipynb`) that demonstrates:

| Use Case | Description |
|---|---|
| **Image Segmentation** | Semantic/instance segmentation (e.g., UNet on COCO/Cityscapes) |
| **Image Classification** | CNN/ViT classifier (e.g., ResNet on CIFAR-10/ImageNet subset) |
| **Image Detection** | Object detection (e.g., YOLO/Faster R-CNN) |
| **Video Classification** | Video understanding (e.g., 3D-CNN/TimeSformer on Kinetics subset) |
| **LLM Fine-tuning** | Fine-tune Qwen 3 4B with LoRA/QLoRA |
| **Image Generation** | Text-to-image diffusion model fine-tuning/inference |

## W&B Features to Cover in Each Notebook

Every notebook should demonstrate as many of these W&B features as applicable:

- **Experiment Tracking** (`wandb.init`, `wandb.log`, `wandb.config`) — metrics, system metrics, media logging
- **Artifacts** (`wandb.Artifact`) — dataset versioning, model versioning, lineage tracking
- **Model Registry** — model linking, aliases (staging/production), lifecycle management
- **Sweeps** (`wandb.sweep`, `wandb.agent`) — hyperparameter optimization
- **Reports** — programmatic report creation via `wandb.apis.reports`
- **Tables** (`wandb.Table`) — data visualization, prediction comparison
- **Automations** — webhook triggers on model registry events (e.g., auto-deploy on "production" alias)
- **Launch** — job creation and execution (if applicable)

## Automations & Deployment Pipeline

- W&B Automations triggers a webhook when a model version is promoted to "production" alias in Model Registry
- The webhook kicks off a deployment workflow (e.g., containerization or model server update)
- A **Streamlit app** (`models/app/`) displays the deployed model status and allows inference testing

## Notebook Conventions

- All notebooks must be **Colab-ready**: include `!pip install` cells at the top
- First cell should install dependencies: `!pip install wandb torch torchvision transformers ...`
- Second cell should handle W&B login: `wandb.login()` (Colab will prompt for API key)
- Use `wandb.init(project="wandb-e2e-demo-<use_case>")` for consistent project naming
- Include markdown cells with clear explanations in Korean (한국어) for demo narration
- Each notebook should be self-contained and runnable top-to-bottom without external state

## Tech Stack

- **Python 3.10+**
- **PyTorch** as the primary deep learning framework
- **Hugging Face Transformers / PEFT** for LLM fine-tuning
- **Diffusers** for image generation
- **W&B SDK** (`wandb`) for all tracking and platform features
- **Streamlit** for the deployment viewer app
- **Google Colab** as the execution environment

## Key W&B API Patterns

```python
# Standard init
run = wandb.init(project="wandb-e2e-demo-<usecase>", config={...})

# Log metrics
wandb.log({"loss": loss, "accuracy": acc, "epoch": epoch})

# Log media (images, video, etc.)
wandb.log({"predictions": wandb.Image(img, caption="pred")})

# Artifact versioning
artifact = wandb.Artifact("dataset-name", type="dataset")
artifact.add_dir("./data")
run.log_artifact(artifact)

# Model registry linking
run.link_artifact(model_artifact, "model-registry/<model-name>", aliases=["staging"])

# Sweep
sweep_id = wandb.sweep(sweep_config, project="wandb-e2e-demo-<usecase>")
wandb.agent(sweep_id, function=train, count=10)
```
