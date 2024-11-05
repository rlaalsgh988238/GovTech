import pandas as pd
import matplotlib.pyplot as plt

# 데이터 파일 불러오기
apple_df = pd.read_csv('../apple_honglo/apple_preprocessed.csv')
cpi_df = pd.read_csv('../ consumer_price_index/daily_consumer_price_index_preprocessed.csv')
fresh_food_df = pd.read_csv('../fresh_food_index/fresh_food_index.csv')
rainfall_df = pd.read_csv('../RainData/rainfall_daily_avg.csv')
oil_df = pd.read_csv('../OilData/output.csv')

# 날짜 열을 datetime 형식으로 변환
apple_df['날짜'] = pd.to_datetime(apple_df['날짜'])
cpi_df['날짜'] = pd.to_datetime(cpi_df['날짜'])
fresh_food_df['날짜'] = pd.to_datetime(fresh_food_df['날짜'])
rainfall_df['날짜'] = pd.to_datetime(rainfall_df['날짜'])
oil_df['날짜'] = pd.to_datetime(oil_df['날짜'])

# 사과 데이터의 날짜 범위 확인
start_date = apple_df['날짜'].min()
end_date = apple_df['날짜'].max()

# 각 데이터프레임을 사과 데이터의 날짜 범위로 필터링
cpi_filtered = cpi_df[(cpi_df['날짜'] >= start_date) & (cpi_df['날짜'] <= end_date)]
fresh_food_filtered = fresh_food_df[(fresh_food_df['날짜'] >= start_date) & (fresh_food_df['날짜'] <= end_date)]
rainfall_filtered = rainfall_df[(rainfall_df['날짜'] >= start_date) & (rainfall_df['날짜'] <= end_date)]
oil_filtered = oil_df[(oil_df['날짜'] >= start_date) & (oil_df['날짜'] <= end_date)]

# 필터링된 데이터 병합
merged_df = apple_df.merge(cpi_filtered, on='날짜', how='left')
merged_df = merged_df.merge(fresh_food_filtered, on='날짜', how='left')
merged_df = merged_df.merge(rainfall_filtered, on='날짜', how='left')
merged_df = merged_df.merge(oil_filtered, on='날짜', how='left')

# 결과 확인
print(merged_df.head())
print(merged_df.columns)
print(merged_df.info())

# 결과 저장
merged_df.to_csv('merged_apple_data.csv', index=False)

# 데이터 시각화
plt.figure(figsize=(12, 6))
plt.plot(merged_df['날짜'], merged_df['가격(원)'], label='사과 가격')
plt.plot(merged_df['날짜'], merged_df['전국'], label='소비자물가지수')
plt.title('사과 가격과 소비자물가지수 추이')
plt.xlabel('날짜')
plt.ylabel('가격 / 지수')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()