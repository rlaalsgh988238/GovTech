import pandas as pd
from google.cloud import firestore
from google.oauth2 import service_account

# 서비스 계정 키 파일 경로
service_account_path = r"/Users/minhokim/Documents/Projects/doc/disabled_toilet/keyfile/dreamhyoja-6bb2237dea02.json"

# Firestore 클라이언트 초기화
credentials = service_account.Credentials.from_service_account_file(service_account_path)
db = firestore.Client(credentials=credentials)

# CSV 파일 경로
csv_file_path = r"/Users/minhokim/Documents/Projects/doc/disabled_toilet/화장실 정보/bYELFPGWyH.csv"

# CSV 파일 읽기
data = pd.read_csv(csv_file_path)

# 데이터 정제 함수
def clean_data(doc):
    cleaned_doc = {}
    for key, value in doc.items():
        # 빈 문자열 또는 None 값을 기본값으로 대체
        if pd.isna(value) or value == "":
            cleaned_doc[key] = "N/A"  # 기본값으로 "N/A" 사용
        else:
            cleaned_doc[key] = value
    return cleaned_doc

# Firestore 컬렉션에 데이터 업로드
for index, row in data.iterrows():
    try:
        # 문서 데이터 정제
        cleaned_doc = clean_data(row)
        # Firestore 컬렉션에 문서 추가
        db.collection('dreamhyoja').add(cleaned_doc)
        print('Document added:', cleaned_doc)
    except Exception as e:
        # 문서 추가 중 오류 발생 시 예외 처리
        print('Error adding document:', e)
