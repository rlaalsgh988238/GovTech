import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

# 폰트 설정 추가
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 읽기
train_df = pd.read_csv('../../filtering/noyear.csv')

# 날짜 열을 datetime 형식으로 변환 후 정렬
train_df['날짜'] = pd.to_datetime(train_df['날짜'])
train_df = train_df.sort_values(by='날짜')

# 날짜 열 제외
train_x = train_df.drop(columns=['가격(원)','날짜','신선과실지수','사과_지수', '전국_소비자지수'])
train_y = train_df['가격(원)']

# 학습용, 테스트용 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# 랜덤 포레스트 회귀 모델 정의 및 학습
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

# 예측 및 평가
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R² Score:", rf_regressor.score(X_test, y_test))

# 날짜를 x축으로 사용하기 위해 테스트 데이터셋에 있는 날짜 열 가져오기
dates = train_df.loc[X_test.index, '날짜']

# 테스트 데이터와 예측값을 날짜 순서대로 정렬
sorted_indices = dates.argsort()
dates = dates.iloc[sorted_indices]
y_test_sorted = y_test.iloc[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# 예측값과 실제값을 데이터프레임으로 만들기
results = pd.DataFrame({
    '날짜': dates,
    '실제값': y_test_sorted,
    '예측값': y_pred_sorted
})

# 결과 출력
print(results)

# 그래프 설정
plt.figure(figsize=(15, 10))

# 1. 예측값 vs 실제값 그래프
plt.subplot(2, 1, 1)
plt.plot(dates, y_test_sorted, label='실제값', color='blue')
plt.plot(dates, y_pred_sorted, label='예측값', color='red', linestyle='--')
plt.xlabel('날짜')
plt.ylabel('가격(원)')
plt.title('예측값 vs 실제값')
plt.legend()
plt.xticks(rotation=45)

# 2. 특성 중요도 막대 그래프
importance = pd.DataFrame({
    'feature': train_x.columns,
    'importance': rf_regressor.feature_importances_
})
importance = importance.sort_values('importance', ascending=False)

plt.subplot(2, 1, 2)
sns.barplot(x='importance', y='feature', data=importance)
plt.title('특성 중요도')
plt.xlabel('중요도')

plt.tight_layout()
plt.show()
