import uvicorn
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = "./dr.h5"
model = load_model(model_path)

class_labels = {
    0: "normal",
    1: "mild",
    2: "moderate",
    3: "severe",
    4: "proliferative"
}

@app.post("/")
def index():
    return {'message':'Online'}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = Image.open(file.file).convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image)
    image = preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    result = class_labels[predicted_class]
    return {"result": result}

if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=int(os.environ.get("PORT", 8000)))
