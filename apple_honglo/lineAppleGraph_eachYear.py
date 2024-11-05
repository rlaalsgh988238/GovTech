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
all_dates = pd.date_range(start='2000-08-01', end='2000-10-31')
all_month_days = all_dates.strftime('%m-%d').tolist()

# 15일 간격의 x축 눈금 생성
xticks_indices = list(range(0, len(all_month_days), 15))
xticks_labels = [all_month_days[i] for i in xticks_indices]

# 서브플롯 생성 (가로로 나열)
n_years = len(years)
fig, axs = plt.subplots(1, n_years, figsize=(5*n_years, 6), sharey=True)
fig.suptitle('Price Line Plot for Grade "special" (June to December) by Year', fontsize=16)

for idx, year in enumerate(years):
    yearly_data = data[data['날짜'].dt.year == year]

    # 모든 월-일에 대해 데이터 준비 (없는 날짜는 NaN으로 채움)
    full_year_data = pd.DataFrame({'월-일': all_month_days})
    full_year_data = full_year_data.merge(yearly_data[['월-일', '가격(원)']], on='월-일', how='left')

    axs[idx].plot(full_year_data['월-일'], full_year_data['가격(원)'], label=f'{year}')
    axs[idx].set_title(f'Year {year}')
    axs[idx].set_xlabel('Month-Day')
    axs[idx].grid(True, linestyle='--', alpha=0.7)
    axs[idx].legend()

    # x축 눈금 설정
    axs[idx].set_xticks(xticks_indices)
    axs[idx].set_xticklabels(xticks_labels, rotation=45)

# y축 레이블 설정 (첫 번째 서브플롯에만)
axs[0].set_ylabel('Price (Won)')

# 레이아웃 조정
plt.tight_layout()
plt.show()