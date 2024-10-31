from fastapi import FastAPI, HTTPException, Request, Depends
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
from sqlalchemy.orm import Session
from database import SessionLocal, engine
from models import Food
from s3_upload_handler import upload_image_to_s3
from fastapi.responses import JSONResponse
from exception_handler import CustomException, custom_exception_handler
from res_code import ResCode
import yaml

# FastAPI app 생성
app = FastAPI()

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 모델 경로 설정
MODELS_PATH = "./models"

# 설정 파일 로드
with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)
CONFIDENCE_THRESHOLD = config.get("confidence_threshold", 0.7)

# 모델 불러오기
logger.info("Loading models...")
ort_efficient = onnxruntime.InferenceSession(os.path.join(MODELS_PATH, "efficientnet_best_model.onnx"))
model_yolo = YOLO(os.path.join(MODELS_PATH, "best.pt"))
logger.info("Models loaded successfully.")

# 예외 핸들러 등록
app.add_exception_handler(CustomException, custom_exception_handler)

# 데이터베이스 세션 의존성 생성
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

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

# 반환 DTO 정의
class FoodDto(BaseModel):
    foodName: str
    foodImgUrl: str
    calorie: int
    protein: float
    carbohydrate: float
    fat: float

class PredictionResponse(BaseModel):
    num_of_food_detected: int
    food: List[FoodDto]

# 음식 예측 API 엔드포인트 정의
@app.post("/predict", response_model=PredictionResponse)
async def predict_food(request: ImageUrl, db: Session = Depends(get_db)):
    try:
        # 이미지 URL에서 이미지 다운로드
        image_url = request.imgUrl
        logger.info(f"Downloading image from URL: {image_url}")
        image_response = requests.get(image_url)
        if image_response.status_code != 200:
            logger.error("Failed to download image.")
            raise CustomException(ResCode.NOT_FOUND)
                    
        image_data = image_response.content
        image = Image.open(BytesIO(image_data))
        logger.info("Image downloaded successfully.")

        # YOLO로 객체 감지
        logger.info("Detecting objects in image using YOLO...")
        detected_objects = model_yolo(image)
        logger.info(f"Number of objects detected: {len(detected_objects[0].boxes)}")

        # dish 또는 food가 감지되지 않은 경우 예외 처리
        if not any(model_yolo.names[int(box.cls)] in ['dish', 'food'] for box in detected_objects[0].boxes):
            logger.error("No dish or food detected in the image.")
            raise CustomException(ResCode.DISH_NOT_DETECTED)

        # 감지된 객체 정보 로깅 및 우선 순위에 따라 dish 또는 food 사용
        food_results = []
        selected_boxes = [box for box in detected_objects[0].boxes if model_yolo.names[int(box.cls)] == 'dish']
        if not selected_boxes:
            selected_boxes = [box for box in detected_objects[0].boxes if model_yolo.names[int(box.cls)] == 'food']

        for idx, box in enumerate(selected_boxes):
            logger.info(f"Object {idx + 1} - Box coordinates: {box.xyxy[0]}")

            # 각 감지된 객체 크롭
            logger.info(f"Processing detected object {idx + 1}...")
            cropped_image = crop_food_object(image, box.xyxy[0])
            cropped_image_data = BytesIO()
            cropped_image.save(cropped_image_data, format='JPEG')
            cropped_image_data = cropped_image_data.getvalue()

            # S3에 크롭된 이미지 업로드
            folder_name = "FOOD" if model_yolo.names[int(box.cls)] == 'food' else "DISH"
            image_url = upload_image_to_s3(cropped_image_data, f"{folder_name}/cropped_{idx + 1}.jpg")
            if not image_url:
                logger.error("Failed to upload image to S3.")
                raise CustomException(ResCode.IMAGE_UPLOAD_FAILED)

            # 이미지 전처리
            input_image = preprocess_image(cropped_image_data)
            logger.info(f"Preprocessed image tensor shape: {input_image.shape}")

            # ONNX 모델에 입력하여 추론
            logger.info("Running inference on cropped image using EfficientNet...")
            input_name = ort_efficient.get_inputs()[0].name
            result = ort_efficient.run(None, {input_name: input_image.numpy()})
            result = torch.tensor(result[0])
            probabilities = torch.nn.functional.softmax(result[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            prediction_probability = probabilities[predicted_class].item()
            logger.info(f"Predicted class for object {idx + 1}: {predicted_class} with probability {prediction_probability}")

            # 확률이 설정 파일에 정의된 임계값 미만일 경우 예외 처리
            if prediction_probability < CONFIDENCE_THRESHOLD:
                logger.error(f"Low confidence in prediction for object {idx + 1}: {prediction_probability}")
                raise CustomException(ResCode.LOW_CONFIDENCE_PREDICTION)

            # 데이터베이스에서 음식 정보 조회
            food_info = db.query(Food).filter(Food.class_id == predicted_class).first()
            if not food_info:
                logger.error(f"Food with class ID {predicted_class} not found in database.")
                raise CustomException(ResCode.FOOD_NOT_FOUND)

            # 추론 결과 반환
            food_results.append(FoodDto(
                foodName=food_info.name,
                foodImgUrl=image_url,
                calorie=int(food_info.calorie),
                protein=food_info.protein,
                carbohydrate=food_info.carbohydrate,
                fat=food_info.fat
            ))

        response_data = PredictionResponse(
            num_of_food_detected=len(food_results),
            food=food_results
        )
        logger.info("Prediction completed successfully.")
        return response_data

    except CustomException as e:
        logger.error(f"An error occurred: {str(e)}")
        raise e
    except Exception as e:
        raise e

# FastAPI 실행
def start():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    start()
