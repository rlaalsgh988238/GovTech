import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('noyear_with_previous_price.csv')

# '가격(원)' 열 삭제
df_without_price = df.drop(columns=['가격(원)'])

# 결과를 새로운 CSV 파일로 저장
df_without_price.to_csv('noyear_without_price.csv', index=False, encoding='utf-8-sig')
print('파일 저장 완료: noyear_without_price.csv')
