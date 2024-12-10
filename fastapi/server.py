from fastapi import FastAPI, UploadFile, File
import uvicorn
from pydantic import BaseModel
import requests
from io import BytesIO
import prediction

app = FastAPI()
class PredictRequest(BaseModel):
    url: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict")
async def predict(request: PredictRequest):
    response = requests.get(request.url)
    image = prediction.read_image(response)
    image = prediction.preprocess_image(image)
    predict = prediction.predict_image(image)
    print(f"Image URL: {request.url}, Predicted: {predict}")
    return predict

@app.post("/predict-file")
async def predict(file: UploadFile = File(...)):
    image = prediction.read_image(await file.read())
    image = prediction.preprocess_image(image)
    predict = prediction.predict_image(image)
    print(predict)
    return predict

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)