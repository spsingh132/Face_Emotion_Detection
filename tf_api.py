from fastapi import FastAPI, File, UploadFile
from starlette.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io

# ----------------------------
# Load Model
# ----------------------------
MODEL_PATH = "model_0.h5"
model = load_model(MODEL_PATH)

# Define emotion labels (adjust as per your model training)
EMOTIONS = ['Angry', 'Happy', 'Sad']

# ----------------------------
# Initialize FastAPI
# ----------------------------
app = FastAPI(title="Human Emotion Detection API", version="1.0")

# ----------------------------
# Image Preprocessing
# ----------------------------
def preprocess_image(image: Image.Image):
    image = image.convert('RGB')  # Convert to grayscale if your model expects grayscale
    image = image.resize((224, 224))  # Resize to model input size
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    return img_array

# ----------------------------
# Home Endpoint
# ----------------------------
@app.get("/")
async def home():
    return {"message": "Welcome to the Human Emotion Detection API"}

# ----------------------------
# Predict Endpoint
# ----------------------------
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict emotion
        prediction = model.predict(processed_image)
        emotion_index = np.argmax(prediction)
        emotion = EMOTIONS[emotion_index]
        confidence = float(prediction[0][emotion_index]) * 100
        
        print("Processed shape:", processed_image.shape)
        predictions = model.predict(processed_image)
        print("Predictions shape:", predictions.shape)
        print("Predictions:", predictions)


        return JSONResponse(content={
            "emotion": emotion,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
