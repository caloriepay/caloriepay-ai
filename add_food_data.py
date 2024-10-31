import pandas as pd
from sqlalchemy.orm import Session
from database import SessionLocal
from models import Food

# CSV 파일 경로 설정
csv_file_path = "./food_db.csv"

# CSV 파일 로드 (인코딩 문제 해결을 위해 utf-8 사용)
food_data = pd.read_csv(csv_file_path, encoding='utf-8')

# 음식 데이터 추가 함수
def add_food_data():
    db = SessionLocal()
    try:
        # food_data의 각 행을 반복하여 데이터베이스에 추가
        for _, row in food_data.iterrows():
            food = Food(
                class_id=int(row['class_id']),
                name=row['name'],
                calorie=float(row['calorie']),
                protein=float(row['protein']),
                carbohydrate=float(row['carbohydrate']),
                fat=float(row['fat'])
            )
            db.add(food)
        db.commit()
        print("Food data added successfully!")
    except Exception as e:
        db.rollback()
        print(f"Error occurred: {str(e)}")
    finally:
        db.close()

if __name__ == "__main__":
    add_food_data()
