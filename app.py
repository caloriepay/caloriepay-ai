from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List
import onnxruntime
import torch
import requests
from PIL import Image
import torchvision.transforms as transforms
from ultralytics import YOLO
import numpy as np
from io import BytesIO
import uvicorn
import os
import logging

# FastAPI app 생성
app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 경로 설정
MODELS_PATH = "./models"

# 모델 불러오기
logger.info("Loading models...")
ort_efficient = onnxruntime.InferenceSession(os.path.join(MODELS_PATH, "efficientnet_best_model.onnx"))
model_yolo = YOLO(os.path.join(MODELS_PATH, "best.pt"))
logger.info("Models loaded successfully.")

# 이미지 전처리 함수
def preprocess_image(image_data):
    logger.info("Preprocessing image...")
    image = Image.open(BytesIO(image_data))
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # 모델에 들어갈 수 있는 배치 차원 추가
    logger.info("Image preprocessed.")
    return input_batch

# Yolov5로 감지된 영역 크롭 함수
def crop_food_object(image, box):
    logger.info(f"Cropping image with box coordinates: {box}")
    x1, y1, x2, y2 = map(int, box[:4])
    cropped_image = image.crop((x1, y1, x2, y2))
    logger.info("Image cropped.")
    return cropped_image

# 추론 요청 모델 정의
class ImageUrl(BaseModel):
    imgUrl: str

# 음식 예측 API 엔드포인트 정의
@app.post("/predict")
async def predict_food(request: ImageUrl):
    try:
        # 이미지 URL에서 이미지 다운로드
        image_url = request.imgUrl
        logger.info(f"Downloading image from URL: {image_url}")
        image_response = requests.get(image_url)
        if image_response.status_code != 200:
            logger.error("Failed to download image.")
            raise HTTPException(status_code=400, detail="이미지를 다운로드할 수 없습니다.")

        image_data = image_response.content
        image = Image.open(BytesIO(image_data))
        logger.info("Image downloaded successfully.")

        # YOLO로 객체 감지
        logger.info("Detecting objects in image using YOLO...")
        detected_objects = model_yolo(image)
        logger.info(f"Number of objects detected: {len(detected_objects[0].boxes)}")
        food_results = []

        for obj in detected_objects[0].boxes:
            # 각 감지된 객체 크롭
            logger.info("Processing detected object...")
            cropped_image = crop_food_object(image, obj.xyxy[0])
            cropped_image_data = BytesIO()
            cropped_image.save(cropped_image_data, format='JPEG')
            cropped_image_data = cropped_image_data.getvalue()

            # 이미지 전처리
            input_image = preprocess_image(cropped_image_data)

            # ONNX 모델에 입력하여 추론
            logger.info("Running inference on cropped image using EfficientNet...")
            input_name = ort_efficient.get_inputs()[0].name
            result = ort_efficient.run(None, {input_name: input_image.numpy()})
            result = torch.tensor(result[0])
            probabilities = torch.nn.functional.softmax(result[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            logger.info(f"Predicted class: {predicted_class}")

            # 추론 결과 반환
            food_results.append({
                "predicted_class": predicted_class,
                "food_name": class_dict[str(predicted_class)]
            })

        response_data = {
            'num_of_food_detected': len(food_results),
            'food': food_results
        }
        logger.info("Prediction completed successfully.")
        return response_data

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 음식 클래스 딕셔너리 정의
class_dict = {
    "0": "Ssalbap",
    "1": "Kimchi Fried Rice",
    "2": "Fried Rice",
    "3": "Bibimbap",
    "4": "Omurice",
    "5": "Yukhoe Bibimbap",
    "6": "Curry Rice",
    "7": "Pork Gukbap",
    "8": "Gimbap",
    "9": "Ramen",
    "10": "Mul Naengmyeon",
    "11": "Bibim Naengmyeon",
    "12": "Jajangmyeon",
    "13": "Jjamppong",
    "14": "Seafood Kalguksu",
    "15": "Meat Dumplings",
    "16": "Fried Dumplings",
    "17": "Miyeok Guk",
    "18": "Samgyetang",
    "19": "Doenjang Jjigae",
    "20": "Ham Kimchi Jjigae",
    "21": "Beef Bulgogi",
    "22": "Tonkatsu"
}

# FastAPI 실행
def start():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start()
