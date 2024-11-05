import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# 데이터 로드
apple_df = pd.read_csv('AppleData/2000_2024_Apple_Honglo_Data.csv')
oil_df = pd.read_csv('../OilData/output.csv')
rain_df = pd.read_csv('../RainData/daily_avg_rainfall.csv')

# 데이터 전처리
apple_df['날짜'] = pd.to_datetime(apple_df['날짜'])
oil_df['날짜'] = pd.to_datetime(oil_df['날짜'])
rain_df['일시'] = pd.to_datetime(rain_df['일시'])

# 사과 데이터에서 '상' 등급만 선택하고 10kg로 가격 조정
apple_df = apple_df[apple_df['등급'] == '상']
apple_df['가격(원)'] = apple_df['가격(원)'] * (10 / apple_df['중량(kg)'])

# 날짜를 인덱스로 설정
apple_df.set_index('날짜', inplace=True)
oil_df.set_index('날짜', inplace=True)
rain_df.set_index('일시', inplace=True)

# 중복 제거
apple_df = apple_df[~apple_df.index.duplicated(keep='first')]
oil_df = oil_df[~oil_df.index.duplicated(keep='first')]
rain_df = rain_df[~rain_df.index.duplicated(keep='first')]

# 데이터 병합
merged_df = pd.concat([apple_df['가격(원)'], oil_df['한화'], rain_df['total_rainfall']], axis=1, join='outer')
merged_df.columns = ['apple_price', 'oil_price', 'rainfall']

# 결측치 처리
merged_df = merged_df.interpolate().ffill().bfill()

# 이상치 제거
def remove_outliers(df):
    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[column] = df[column].clip(lower_bound, upper_bound)
    return df

merged_df = remove_outliers(merged_df)

# 스케일링
scaler = RobustScaler()
scaled_data = scaler.fit_transform(merged_df)

# 시퀀스 데이터 생성
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:(i + seq_length), :])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

seq_length = 30
X, y = create_sequences(scaled_data, seq_length)

# 훈련 데이터와 테스트 데이터 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 구성
model = Sequential([
    LSTM(64, activation='relu', input_shape=(seq_length, 3), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# 컴파일
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')

# 조기 종료
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# 모델 훈련
history = model.fit(
    X_train, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

# 2024년 10월 1일 예측
prediction_date = pd.to_datetime('2024-10-01')
last_30_days = merged_df.loc[:prediction_date].iloc[-30:].values
last_30_days_scaled = scaler.transform(last_30_days)

# 임의의 강수량과 원유 가격 생성
random_rainfall = np.random.uniform(0, 100)
random_oil_price = np.random.uniform(80000, 100000)

# 예측 데이터 생성
prediction_data = np.concatenate([last_30_days_scaled[1:],
                                  [[0, scaler.transform([[0, random_oil_price, random_rainfall]])[0][1],
                                    scaler.transform([[0, random_oil_price, random_rainfall]])[0][2]]]])

# 예측
prediction_scaled = model.predict(prediction_data.reshape(1, seq_length, 3))
prediction = scaler.inverse_transform(np.concatenate([prediction_scaled, [[0, 0]]], axis=1))[0][0]

print(f"2024년 10월 1일 예측 사과 가격 (10kg): {prediction:.2f}원")
print(f"사용된 강수량: {random_rainfall:.2f}")
print(f"사용된 원유 가격: {random_oil_price:.2f}")

# 학습 곡선 그리기
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
