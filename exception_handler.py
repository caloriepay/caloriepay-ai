from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse

# 커스텀 예외 정의
class CustomException(HTTPException):
    def __init__(self, res_code: dict):
        super().__init__(status_code=res_code["status_code"], detail=res_code["detail"])
        self.error_code = res_code["error_code"]

# 전역 예외 핸들러 정의
async def custom_exception_handler(request: Request, exc: CustomException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail,
            "error_code": exc.error_code
        }
    )
