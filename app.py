from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = FastAPI()

# Load your model once
model = tf.keras.models.load_model("soil_classifier_efficientnet__better_newK.keras")

# Preprocess image for model
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((224, 224))  # adjust to your model's input size
    return np.expand_dims(np.array(image) / 255.0, axis=0)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img_array = preprocess_image(img_bytes)
    prediction = model.predict(img_array)
    soil_type = int(np.argmax(prediction, axis=1)[0])  # adjust if you have class names
    return JSONResponse(content={"soil_type": soil_type})
