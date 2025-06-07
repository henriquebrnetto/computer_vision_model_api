from ultralytics import YOLO
from PIL import Image, ImageDraw
import io
import base64

def predict_label(image: Image.Image, model_path: str, conf: float = 0.25):
    model = YOLO(model_path)
    results = model.predict(source=image, conf=conf)[0]

    detections = []
    annotated = image.convert("RGB")
    draw = ImageDraw.Draw(annotated)

    for box, cls, conf_score in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        label = model.names[int(cls)]
        confidence = float(conf_score)

        # Guarda valores
        detections.append({
            "label": label,
            "confidence": round(confidence, 3),
            "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]
        })

        # Desenha retângulo e texto
        draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
        draw.text((x1, y1 - 10), f"{label} {confidence:.2f}", fill="red")

    # Converte imagem anotada para Base64
    buffered = io.BytesIO()
    annotated.save(buffered, format="JPEG")
    encoded = base64.b64encode(buffered.getvalue()).decode()

    return {"predictions": detections, "annotated_image": encoded}