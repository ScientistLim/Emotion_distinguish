from fastapi import FastAPI, UploadFile, File
from main import predict_expression, face_mesh
from fastapi.responses import FileResponse
import cv2 as cv
import numpy as np
import shutil

app = FastAPI()

@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    file_contents = await file.read()
    

    nparr = np.fromstring(file_contents, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    
    expression = predict_expression(image, face_mesh)
    
    return {"emotion": expression}



@app.post("/analyze/")
async def analyze_image(file: UploadFile = File(...)):
    file_contents = await file.read()
    
    nparr = np.fromstring(file_contents, np.uint8)
    image = cv.imdecode(nparr, cv.IMREAD_COLOR)
    
    expression = predict_expression(image, face_mesh)
    
    # Save the uploaded image temporarily
    temp_image_path = "temp_image.jpg"
    cv.imwrite(temp_image_path, image)
    
    return {
        "emotion": expression,
   }