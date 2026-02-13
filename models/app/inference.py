import torch
from PIL import Image
from torchvision import transforms

# CIFAR-10 전용 정규화 값
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def run_inference(model, uploaded_file, metadata: dict) -> dict:
    """모델 유형에 따라 적절한 추론 함수를 호출한다."""
    image = Image.open(uploaded_file).convert("RGB")
    model_type = metadata.get("model_type", "classification")

    if model_type == "classification":
        return _classify(model, image, metadata)
    elif model_type == "detection":
        raise NotImplementedError("Detection inference는 아직 구현되지 않았습니다.")
    elif model_type == "segmentation":
        raise NotImplementedError("Segmentation inference는 아직 구현되지 않았습니다.")
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _classify(model, image: Image.Image, metadata: dict) -> dict:
    """CIFAR-10 classification 추론."""
    classes = metadata.get("classes", [f"Class {i}" for i in range(10)])
    input_size = metadata.get("input_size", [3, 32, 32])
    h, w = input_size[1], input_size[2]

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    model.eval()
    with torch.no_grad():
        output = model(transform(image).unsqueeze(0))
        probs = torch.softmax(output, dim=1)[0]
        top5 = torch.topk(probs, min(5, len(classes)))

    predictions = {
        classes[i.item()]: f"{p.item():.2%}"
        for p, i in zip(top5.values, top5.indices)
    }

    return {
        "visualization": image,
        "predictions": predictions,
        "top_class": classes[top5.indices[0].item()],
        "top_confidence": top5.values[0].item(),
    }
