import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator, DateFormatter


plt.rcParams['font.family'] ='AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 읽기
df = pd.read_csv('AppleData/2000_2024_Apple_Honglo_Data.txt', parse_dates=['날짜'])

# 연도와 등급별로 평균 가격 계산
yearly_avg = df.groupby([df['날짜'].dt.year, '등급'])['가격(원)'].mean().unstack()

# 그래프 설정
plt.figure(figsize=(15, 8))

# 각 등급별로 선 그래프 그리기
for grade in yearly_avg.columns:
    plt.plot(yearly_avg.index, yearly_avg[grade], label=grade, marker='o')

# 그래프 꾸미기
plt.title('연도별 평균 가격 (등급별)', fontsize=16)
plt.xlabel('연도', fontsize=12)
plt.ylabel('평균 가격 (원)', fontsize=12)
plt.legend(title='등급', loc='upper left')

# x축 설정
plt.gca().xaxis.set_major_locator(YearLocator())
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
plt.gcf().autofmt_xdate()  # 날짜 레이블 회전

# y축 눈금 설정
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# 그리드 추가
plt.grid(True, linestyle='--', alpha=0.7)

# 그래프 저장 및 표시
plt.tight_layout()
plt.savefig('yearly_price_by_grade.png', dpi=300)
plt.show()
