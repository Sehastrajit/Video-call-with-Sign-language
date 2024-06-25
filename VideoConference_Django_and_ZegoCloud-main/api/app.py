

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from PIL import Image
from io import BytesIO
import numpy as np
from ultralytics import YOLO
from fastapi.responses import JSONResponse

app = FastAPI()

# Load YOLOv8 model
model = YOLO("best.pt")

class PredictionSchema(BaseModel):
    label: str
    confidence: float
    xmin: int
    ymin: int
    xmax: int
    ymax: int

@app.post("/predict/", response_model=List[PredictionSchema])
async def predict(image: UploadFile = File(...)):
    # Process the uploaded image using YOLOv8
    try:
        contents = await image.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = np.array(img)

        # Perform inference with YOLOv8
        results = model(img_array)

        predictions = []
        for label, confidence, bbox in results.xyxy[0]:
            xmin, ymin, xmax, ymax = map(int, bbox)
            predictions.append({
                "label": label,
                "confidence": float(confidence),
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax
            })

        return predictions
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error processing image: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
