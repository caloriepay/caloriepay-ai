class ResCode:
    OK = {"status_code": 200, "detail": "요청 성공", "error_code": "STATUS-OK"}
    BAD_REQUEST = {"status_code": 400, "detail": "요청이 올바르지 않습니다.", "error_code": "ERROR-BR-000"}
    NOT_FOUND = {"status_code": 404, "detail": "해당 요청 정보를 찾을 수 없습니다.", "error_code": "ERROR-NF-000"}
    INTERNAL_SERVER_ERROR = {"status_code": 500, "detail": "내부 서버 오류입니다. 관리자에게 문의하세요.", "error_code": "ERROR-ISE-000"}
    INVALID_IMAGE_FORMAT = {"status_code": 400, "detail": "지원하지 않는 이미지 형식입니다.", "error_code": "ERROR-BR-010"}
    IMAGE_SIZE_EXCEEDED = {"status_code": 400, "detail": "이미지 파일 크기가 초과되었습니다.", "error_code": "ERROR-BR-011"}
    IMAGE_UPLOAD_FAILED = {"status_code": 400, "detail": "이미지 업로드에 실패했습니다.", "error_code": "ERROR-BR-012"}
    DISH_NOT_DETECTED = {"status_code": 404, "detail": "음식이 탐지되지 않았습니다.", "error_code": "ERROR-NF-002"}
    FOOD_NOT_FOUND = {"status_code": 404, "detail": "해당 음식 정보를 찾을 수 없습니다.", "error_code": "ERROR-NF-003"}
    UNAUTHORIZED = {"status_code": 401, "detail": "권한이 없습니다.", "error_code": "ERROR-UA-000"}
    CONFLICT = {"status_code": 409, "detail": "요청에 충돌이 발생했습니다.", "error_code": "ERROR-CF-000"}