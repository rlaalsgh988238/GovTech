import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('apple_preprocessed.csv')

# 등급별로 데이터 분리
grades = df['등급'].unique()

# 각 등급에 대해 별도의 CSV 파일로 저장
for grade in grades:
    grade_df = df[df['등급'] == grade]
    filename = f'apple_{grade}.csv'
    grade_df.to_csv(filename, index=False, encoding='utf-8-sig')
    print(f'파일 저장 완료: {filename}')
