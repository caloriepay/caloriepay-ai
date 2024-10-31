# models.py
from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Food(Base):
    __tablename__ = 'foods'

    id = Column(Integer, primary_key=True, index=True)
    class_id = Column(Integer, unique=True, index=True)  # 예측된 클래스 ID와 매칭되는 필드 추가
    name = Column(String, index=True)
    calorie = Column(Integer)
    protein = Column(Float)
    carbohydrate = Column(Float)
    fat = Column(Float)
