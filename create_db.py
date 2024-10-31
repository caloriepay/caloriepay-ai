# create_db.py
from database import engine
from models import Base

# 데이터베이스와 테이블 생성
Base.metadata.create_all(bind=engine)
