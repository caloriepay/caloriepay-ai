from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import logging

app = FastAPI()
logger = logging.getLogger(__name__)

# 커스텀 예외 클래스 정의
class CustomAPIException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

# 커스텀 예외 핸들러 등록
@app.exception_handler(CustomAPIException)
async def custom_api_exception_handler(request: Request, exc: CustomAPIException):
    logger.error(f"Exception occurred: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "CustomAPIException",
                "message": exc.detail,
                "request": str(request.url)
            }
        }
    )

# FastAPI의 HTTPException 핸들러를 커스터마이징
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.error(f"HTTP Exception occurred: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "type": "HTTPException",
                "message": exc.detail,
                "request": str(request.url)
            }
        }
    )

# 다른 종류의 예외를 처리하는 핸들러 등록
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unexpected error: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "type": "Exception",
                "message": "An unexpected error occurred. Please try again later.",
                "request": str(request.url)
            }
        }
    )

