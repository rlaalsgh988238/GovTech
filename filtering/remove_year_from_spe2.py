import pandas as pd

# CSV 파일 읽기
data = pd.read_csv('apple_data_special2.csv')

# '연' 열 제거
data = data.drop(columns=['연'])

data.to_csv('noyear.csv', index=False)