import pandas as pd

# 두 개의 CSV 파일 읽기
df1 = pd.read_csv('noyear_without_price.csv')
df2 = pd.read_csv('../apple_honglo/apple_상.csv')

# 공통 열을 기준으로 병합 (예: '날짜' 열)
merged_df = pd.merge(df1, df2, on='날짜', how='inner')

# 병합된 결과를 새로운 CSV 파일로 저장
merged_df.to_csv('상_data.csv', index=False, encoding='utf-8-sig')
print('파일 저장 완료: merged_data.csv')
