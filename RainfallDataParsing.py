import pandas as pd
import numpy as np

# 데이터 파일 읽기
df = pd.read_csv('RainData/rainData.txt', parse_dates=['일시'])

# '일강수량(mm)' 열의 문자열을 숫자로 변환하는 함수
def parse_rainfall(rainfall_str):
    try:
        return float(rainfall_str.replace(',', '.'))
    except:
        return np.nan

# '일강수량(mm)' 열 처리
df['rainfall'] = df['일강수량(mm)'].apply(parse_rainfall)

# 일자별 평균 강수량 계산
daily_avg_rainfall = df.groupby('일시')['rainfall'].mean().reset_index()

# 결과 정렬
daily_avg_rainfall = daily_avg_rainfall.sort_values('일시')

# 전체 평균 강수량 계산
overall_avg_rainfall = daily_avg_rainfall['rainfall'].mean()

# 결과 출력
print("일자별 평균 강수량:")
print(daily_avg_rainfall)
print("\n전체 평균 강수량:", overall_avg_rainfall)

# 결과를 CSV 파일로 저장 (선택사항)
daily_avg_rainfall.to_csv('daily_avg_rainfall.csv', index=False)
