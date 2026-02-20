import numpy as np
from PIL import Image


def run_inference(model, uploaded_file, metadata: dict) -> dict:
    """YOLO-seg 모델로 추론하여 classification/detection/segmentation 결과를 반환한다."""
    image = Image.open(uploaded_file).convert("RGB")
    results = model(image, verbose=False)
    r = results[0]
    classes = metadata.get("classes", list(r.names.values()) if r.names else [])

    det_image = Image.fromarray(r.plot(masks=False)[..., ::-1])

    seg_image = None
    if r.masks is not None:
        seg_image = Image.fromarray(r.plot(boxes=False)[..., ::-1])

    detections = []
    class_summary: dict[str, dict] = {}
    for box in r.boxes:
        cls_id = int(box.cls)
        cls_name = classes[cls_id] if cls_id < len(classes) else r.names.get(cls_id, str(cls_id))
        conf = float(box.conf)
        detections.append({
            "클래스": cls_name,
            "신뢰도": f"{conf:.2%}",
        })
        if cls_name in class_summary:
            class_summary[cls_name]["count"] += 1
            class_summary[cls_name]["max_conf"] = max(class_summary[cls_name]["max_conf"], conf)
        else:
            class_summary[cls_name] = {"count": 1, "max_conf": conf}

    class_summary = dict(sorted(class_summary.items(), key=lambda x: x[1]["count"], reverse=True))

    return {
        "original_image": image,
        "det_image": det_image,
        "seg_image": seg_image,
        "detections": detections,
        "class_summary": class_summary,
        "num_objects": len(r.boxes),
    }
