import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('merged_apple_data.csv')

# 특 등급만 필터링
df_special = df[df['등급'] == '특']

# 날짜 열을 datetime 형식으로 변환
df_special['날짜'] = pd.to_datetime(df_special['날짜'])

# 연, 월, 일 열 추가
df_special['연'] = df_special['날짜'].dt.year
df_special['월'] = df_special['날짜'].dt.month
df_special['일'] = df_special['날짜'].dt.day

# 필요한 열만 선택 (등급 열 제외)
columns_to_keep = ['날짜','연', '월', '일', '가격(원)', '전국', '전국_신선과실', '전국_사과', 'total_rainfall', '한화']
df_final = df_special[columns_to_keep]

# 새로운 CSV 파일로 저장
df_final.to_csv('apple_data_special2.csv', index=False)

# 결과 확인
print("데이터 shape:", df_final.shape)
print("\n처음 5개 행:")
print(df_final.head())