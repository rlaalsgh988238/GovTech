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

# 모든 월-일 조합 생성 (6월 1일부터 12월 31일까지)
all_dates = pd.date_range(start='2000-06-01', end='2000-12-31')
all_month_days = all_dates.strftime('%m-%d').tolist()

# 연도별로 그래프 그리기
fig, axes = plt.subplots(len(years), 1, figsize=(12, 5*len(years)), sharex=True)
fig.suptitle('Price Scatter Plot for Grade "special" (June to December)', fontsize=16)

for idx, year in enumerate(years):
    yearly_data = data[data['날짜'].dt.year == year]
    # 모든 월-일에 대해 데이터 준비 (없는 날짜는 NaN으로 채움)
    full_year_data = pd.DataFrame({'월-일': all_month_days})
    full_year_data = full_year_data.merge(yearly_data[['월-일', '가격(원)']], on='월-일', how='left')

    axes[idx].scatter(full_year_data['월-일'], full_year_data['가격(원)'], alpha=0.6)
    axes[idx].set_title(f'Year {year}')
    axes[idx].set_ylabel('Price (Won)')
    axes[idx].grid(True, linestyle='--', alpha=0.7)

    # y축 범위를 모든 그래프에 대해 동일하게 설정
    axes[idx].set_ylim(data['가격(원)'].min(), data['가격(원)'].max())

# x축 레이블 설정 (마지막 그래프에만)
axes[-1].set_xlabel('Month-Day')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()