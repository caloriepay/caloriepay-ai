import boto3
import yaml
import uuid
from botocore.exceptions import NoCredentialsError

# config.yaml 파일에서 설정 로드
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

AWS_ACCESS_KEY = config['aws']['access_key']
AWS_SECRET_KEY = config['aws']['secret_key']
AWS_BUCKET_NAME = config['aws']['bucket_name']
AWS_REGION = config['aws']['region']

s3 = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY, region_name=AWS_REGION)

# S3에 이미지 업로드 함수
def upload_image_to_s3(image_data, filename):
    try:
        unique_filename = f"{uuid.uuid4()}_{filename}"
        s3.put_object(Bucket=AWS_BUCKET_NAME, Key=unique_filename, Body=image_data, ContentType='image/jpeg')
        image_url = f"https://{AWS_BUCKET_NAME}.s3.{AWS_REGION}.amazonaws.com/{unique_filename}"
        return image_url
    except NoCredentialsError:
        print("Credentials not available")
        return None
    except Exception as e:
        print(f"Error occurred during S3 upload: {str(e)}")
        return None
