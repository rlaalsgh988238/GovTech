#밑의 코드는 하루단위로 예측하는건데
#주단위로 예측하게 밑의 코드를 고쳐줘
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc

# 폰트 설정 추가
rc('font', family='AppleGothic')
plt.rcParams['axes.unicode_minus'] = False

# 데이터 읽기
train_df = pd.read_csv('../../filtering/noyear_with_previous_price.csv')

# 날짜 열을 datetime 형식으로 변환 후 정렬
train_df['날짜'] = pd.to_datetime(train_df['날짜'])
train_df = train_df.sort_values(by='날짜')
train_df = train_df.dropna()

# 2023년까지의 데이터로 학습 데이터셋 만들기
train_data = train_df[train_df['날짜'] < '2024-01-01']
test_data = train_df[train_df['날짜'] >= '2024-01-01']

# 날짜 열 제외
train_x = train_data.drop(columns=['가격(원)', '날짜', '신선과실지수', '사과_지수'])
train_y = train_data['가격(원)']


# 학습용, 테스트용 데이터셋 분리
X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# 랜덤 포레스트 회귀 모델 정의 및 학습
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)

# 지정날짜 이후 데이터로 예측
test_x = test_data.drop(columns=['가격(원)', '날짜', '신선과실지수', '사과_지수'])
y_test_actual = test_data['가격(원)']
y_pred_2024 = rf_regressor.predict(test_x)

# MSE와 R² 점수 계산
mse_2024 = mean_squared_error(y_test_actual, y_pred_2024)
r2_2024 = r2_score(y_test_actual, y_pred_2024)

print("2024년 예측 결과")
print("Mean Squared Error:", mse_2024)
print("R² Score:", r2_2024)

# 날짜를 x축으로 사용하기 위해 2024년 데이터의 날짜 열 가져오기
dates_2024 = test_data['날짜']

# 예측값과 실제값을 데이터프레임으로 만들기
results_2024 = pd.DataFrame({
    '날짜': dates_2024,
    '실제값': y_test_actual,
    '예측값': y_pred_2024
})

# 결과 출력
print(results_2024)

# 그래프 설정
plt.figure(figsize=(15, 10))

# 1. 예측값 vs 실제값 그래프
plt.subplot(2, 1, 1)
plt.plot(dates_2024, y_test_actual, label='실제값', color='blue')
plt.plot(dates_2024, y_pred_2024, label='예측값', color='red', linestyle='--')
plt.xlabel('날짜')
plt.ylabel('가격(원)')
plt.title('2024년 예측값 vs 실제값')
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
