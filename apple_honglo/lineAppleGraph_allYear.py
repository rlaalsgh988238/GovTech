import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
data = pd.read_csv('apple_preprocessed.csv')
# 날짜를 datetime 형식으로 변환
data['날짜'] = pd.to_datetime(data['날짜'])

# 6월부터 12월까지의 데이터 필터링 및 '특' 등급만 선택
data = data[(data['날짜'].dt.month >= 6) & (data['날짜'].dt.month <= 12) & (data['등급'] == '특')]

# 월-일 형식으로 변환
data['월-일'] = data['날짜'].dt.strftime('%m-%d')

# 연도별로 데이터 분리
years = sorted(data['날짜'].dt.year.unique())

# 그래프 그리기
plt.figure(figsize=(12, 8))

# 모든 월-일 조합 생성 (6월 1일부터 12월 31일까지)
all_dates = pd.date_range(start='2000-06-01', end='2000-12-31')
all_month_days = all_dates.strftime('%m-%d').tolist()

for year in years:
    yearly_data = data[data['날짜'].dt.year == year]
    # 모든 월-일에 대해 데이터 준비 (없는 날짜는 NaN으로 채움)
    full_year_data = pd.DataFrame({'월-일': all_month_days})
    full_year_data = full_year_data.merge(yearly_data[['월-일', '가격(원)']], on='월-일', how='left')

    plt.plot(full_year_data['월-일'], full_year_data['가격(원)'], label=f'{year}', alpha=0.6)

plt.title('Price Line Plot for Grade "special" (June to December)')
plt.xlabel('Month-Day')
plt.ylabel('Price (Won)')
plt.xticks(rotation=45)
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()