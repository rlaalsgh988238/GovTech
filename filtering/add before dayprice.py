import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('noyear.csv')

# '날짜' 열을 datetime 형식으로 변환
df['날짜'] = pd.to_datetime(df['날짜'])

# '이전날짜_가격' 열 추가
df['이전날짜_가격'] = df['가격(원)'].shift(1)

# NaN 값을 가장 가까운 이전날짜의 가격으로 채우기
df['이전날짜_가격'] = df['이전날짜_가격'].fillna(method='bfill')

# 새로운 CSV 파일로 저장
df.to_csv('noyear_with_previous_price.csv', index=False)
