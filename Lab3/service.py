import bentoml
from ultralytics import YOLO
from bentoml.io import Image, JSON
from PIL import Image as PILImage

service = bentoml.Service('onnx_detector')
onnx_model = YOLO('best_op.onnx')

@service.api(input=Image(), output=JSON())
def detect(image: PILImage.Image):
    results = onnx_model.predict(source=image, save=False, imgsz=416)

    predictions = []
    for result in results:
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            predictions.append({
                "class": result.names[int(cls)],
                "confidence": float(conf),
                "bbox": [float(coord) for coord in box.tolist()]
            })

    return {"predictions": predictions}