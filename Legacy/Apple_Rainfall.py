import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

# 사과 생산량 데이터
apple_production = [1485, 1218, 1284, 1621, 1546, 1843, 1731, 1623, 1430, 1624, 1336, 1502, 1636, 1167]

# 강수량 데이터 (14개의 데이터)
rainfall = [693.4, 1068.2, 780.2, 585, 608.6, 390.9, 454.7, 612.7,
            620.7, 513.4, 1031.4, 613.8, 674.6, 1015.7]

# 연도 (2010년부터 2023년까지 설정, 14개)
years = list(range(2010, 2024))

# 상관계수와 p-value 계산
correlation_coefficient, p_value = pearsonr(apple_production, rainfall)

# 상관계수와 p-value 출력
print(f'Correlation Coefficient: {correlation_coefficient:.2f}')
print(f'p-value: {p_value:.5f}')

# 그래프 생성
plt.figure(figsize=(14, 7))

# 사과 생산량 그래프
plt.plot(years, apple_production, label='Apple Production (kg per 10a)', marker='o', color='blue')

# 강수량 그래프
plt.plot(years, rainfall, label='Rainfall (mm)', marker='o', color='green')

# 그래프 제목과 축 레이블 설정
plt.title('Apple Production and Rainfall Over Time')
plt.xlabel('Year')
plt.ylabel('Value')

# 범례 추가
plt.legend()

# 상관계수와 p-value 텍스트 추가
plt.text(2010.5, max(apple_production) - 50,
         f'Correlation Coefficient: {correlation_coefficient:.2f}\np-value: {p_value:.5f}',
         fontsize=12, color='red', bbox=dict(facecolor='white', alpha=0.5))

# 그래프 표시
plt.grid(True)
plt.show()
