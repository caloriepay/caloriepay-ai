# Dockerfile
FROM python:3.10

COPY . /src
WORKDIR /src

# 필요한 패키지 설치
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \  
    rm -rf /var/lib/apt/lists/*

# 종속성 복사 및 설치
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 설정 파일 및 코드 파일 복사

# Uvicorn으로 FastAPI 애플리케이션 실행
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]