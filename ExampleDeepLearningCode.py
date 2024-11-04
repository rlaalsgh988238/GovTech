import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # 그래프를 그리기 위한 라이브러리 추가
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import matplotlib.pyplot as plt
plt.rcParams['font.family'] ='AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

# 데이터 생성 (강수량, 면적, 최저시급, 비료 가격에 따른 가격)
np.random.seed(42)


# 샘플 데이터 생성
data_size = 1000
rainfall = np.random.uniform(500, 1500, data_size)  # 강수량 (mm)
area = np.random.uniform(10, 100, data_size)  # 면적 (ha)
min_wage = np.random.uniform(8, 15, data_size)  # 최저시급 (USD)
fertilizer_cost = np.random.uniform(50, 200, data_size)  # 비료 가격 (USD)

# 가격 계산
price = (
        0.5 * rainfall
        + 10 * area
        - 100 * min_wage
        + 20 * fertilizer_cost
        + np.random.normal(0, 50, data_size)
)

# 가상의 날짜 생성
dates = pd.date_range(start="2020-01-01", periods=data_size, freq="D")

# 데이터프레임으로 통합
df = pd.DataFrame({
    "Date": dates,
    "Rainfall": rainfall,
    "Area": area,
    "Min_Wage": min_wage,
    "Fertilizer_Cost": fertilizer_cost,
    "Price": price,
})

# 날짜로 정렬
df.sort_values("Date", inplace=True)

# 입력(X)와 출력(y) 분리
X = df[["Rainfall", "Area", "Min_Wage", "Fertilizer_Cost"]].values
y = df["Price"].values

# 데이터 분할 (훈련 80%, 테스트 20%) - 시간 순서대로 분할
train_size = int(len(df) * 0.8)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]
dates_train = df["Date"][:train_size]
dates_test = df["Date"][train_size:]

# 데이터 표준화
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 모델 생성
model = Sequential()
model.add(Dense(64, input_dim=4, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(1))

# 모델 컴파일
model.compile(optimizer="adam", loss="mean_squared_error")

# 모델 학습
history = model.fit(
    X_train_scaled, y_train, epochs=100, batch_size=32, validation_split=0.2
)

# 학습 과정의 손실 값 시각화
plt.figure(figsize=(10, 6))
plt.plot(history.history["loss"], label="훈련 손실")
plt.plot(history.history["val_loss"], label="검증 손실")
plt.title("에포크별 모델 손실")
plt.xlabel("에포크")
plt.ylabel("손실")
plt.legend()
plt.show()

# 테스트 데이터로 모델 평가
loss = model.evaluate(X_test_scaled, y_test)
print(f"테스트 손실: {loss}")

# 예측 예시
predictions = model.predict(X_test_scaled).flatten()

# 실제 값과 예측 값 비교 시각화
plt.figure(figsize=(14, 7))
plt.plot(dates_test, y_test, label="실제 가격")
plt.plot(dates_test, predictions, label="예측 가격")
plt.title("실제 가격과 예측 가격 비교")
plt.xlabel("날짜")
plt.ylabel("가격")
plt.legend()
plt.show()

# 특정 기간(첫 100일) 확대
plt.figure(figsize=(14, 7))
plt.plot(dates_test[:100], y_test[:100], label="실제 가격")
plt.plot(dates_test[:100], predictions[:100], label="예측 가격")
plt.title("실제 가격과 예측 가격 비교 (첫 100일)")
plt.xlabel("날짜")
plt.ylabel("가격")
plt.legend()
plt.show()