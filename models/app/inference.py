import torch
import numpy as np
from PIL import Image
from torchvision import transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

SEG_COLORS = {
    0: (0, 200, 0),    # foreground — 초록
    1: (40, 40, 40),    # background — 어두운 회색
    2: (200, 0, 0),     # boundary — 빨강
}


def run_inference(model, uploaded_file, metadata: dict) -> dict:
    """모델 유형에 따라 적절한 추론 함수를 호출한다."""
    image = Image.open(uploaded_file).convert("RGB")
    model_type = metadata.get("model_type", "classification")

    if model_type == "classification":
        return _classify(model, image, metadata)
    elif model_type == "segmentation":
        return _segment(model, image, metadata)
    elif model_type == "detection":
        return _detect(model, image, metadata)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _classify(model, image: Image.Image, metadata: dict) -> dict:
    """CIFAR-10 classification 추론. 전처리 단계별 시각화 포함."""
    classes = metadata.get("classes", [f"Class {i}" for i in range(10)])
    input_size = metadata.get("input_size", [3, 32, 32])
    h, w = input_size[1], input_size[2]

    # 전처리 단계별 시각화용 이미지 저장
    original_image = image.copy()
    resized_image = image.resize((h, w), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    model.eval()
    with torch.no_grad():
        input_tensor = transform(image)
        output = model(input_tensor.unsqueeze(0))
        probs = torch.softmax(output, dim=1)[0]
        top5 = torch.topk(probs, min(5, len(classes)))

    predictions = {
        classes[i.item()]: f"{p.item():.2%}"
        for p, i in zip(top5.values, top5.indices)
    }

    return {
        "original_image": original_image,
        "resized_image": resized_image,
        "input_size": (h, w),
        "predictions": predictions,
        "top_class": classes[top5.indices[0].item()],
        "top_confidence": top5.values[0].item(),
    }


def _segment(model, image: Image.Image, metadata: dict) -> dict:
    """DeepLabV3 segmentation 추론. 마스크 오버레이 시각화 포함."""
    classes = metadata.get("classes", ["foreground", "background", "boundary"])
    input_size = metadata.get("input_size", [3, 128, 128])
    h, w = input_size[1], input_size[2]

    original_image = image.copy()
    resized_image = image.resize((h, w), Image.BILINEAR)

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    model.eval()
    with torch.no_grad():
        input_tensor = transform(image)
        output = model(input_tensor.unsqueeze(0))["out"]
        mask = output.argmax(1).squeeze(0).cpu().numpy()

    # 컬러 마스크 생성
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in SEG_COLORS.items():
        color_mask[mask == cls_id] = color
    mask_image = Image.fromarray(color_mask)

    # 원본 위에 마스크 오버레이 (알파 블렌딩)
    base = np.array(resized_image, dtype=np.float32)
    overlay_arr = base.copy()
    alpha = 0.45
    for cls_id, color in SEG_COLORS.items():
        region = mask == cls_id
        overlay_arr[region] = base[region] * (1 - alpha) + np.array(color, dtype=np.float32) * alpha
    overlay_image = Image.fromarray(overlay_arr.astype(np.uint8))

    # 클래스별 픽셀 비율
    total_pixels = mask.size
    class_ratios = {}
    for cls_id, cls_name in enumerate(classes):
        ratio = (mask == cls_id).sum() / total_pixels
        class_ratios[cls_name] = f"{ratio:.1%}"

    return {
        "original_image": original_image,
        "resized_image": resized_image,
        "mask_image": mask_image,
        "overlay_image": overlay_image,
        "input_size": (h, w),
        "class_ratios": class_ratios,
        "classes": classes,
    }


def _detect(model, image: Image.Image, metadata: dict) -> dict:
    """YOLOv8 detection 추론. Ultralytics 내장 시각화 활용."""
    classes = metadata.get("classes", [])

    results = model(image, verbose=False)
    r = results[0]

    plotted = Image.fromarray(r.plot()[..., ::-1])  # BGR -> RGB

    detections = []
    for box in r.boxes:
        cls_id = int(box.cls)
        cls_name = classes[cls_id] if cls_id < len(classes) else str(cls_id)
        detections.append({
            "class": cls_name,
            "confidence": f"{float(box.conf):.2%}",
        })

    return {
        "original_image": image,
        "result_image": plotted,
        "detections": detections,
        "num_objects": len(r.boxes),
    }
