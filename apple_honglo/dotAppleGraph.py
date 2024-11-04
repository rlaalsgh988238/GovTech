import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
data = pd.read_csv('apple_preprocessed.csv')
# 날짜를 datetime 형식으로 변환
data['날짜'] = pd.to_datetime(data['날짜'])

# 6월부터 12월까지의 데이터 필터링 및 '특' 등급만 선택
data = data[(data['날짜'].dt.month >= 6) & (data['날짜'].dt.month <= 12) & (data['등급'] == '특')]

# 연도별로 데이터 분리
years = data['날짜'].dt.year.unique()

# 그래프 그리기
plt.figure(figsize=(12, 8))

for year in years:
    yearly_data = data[data['날짜'].dt.year == year]
    # 월-일 형식으로 변환
    dates = yearly_data['날짜'].dt.strftime('%m-%d')
    plt.scatter(
        dates,  # 월-일 형식으로 변환된 날짜
        yearly_data['가격(원)'],
        label=f'{year}',
        alpha=0.6
    )

plt.title('Price Scatter Plot for Grade "high" (June to December)')
plt.xlabel('Month-Day')
plt.ylabel('Price (Won)')
plt.xticks(rotation=45)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
plt.tight_layout()
plt.show()
