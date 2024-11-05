import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 읽기
df = pd.read_csv('typhoon_data.csv')

# 영향도가 0이 아닌 태풍만 선택
df_filtered = df[df['영향도'] != 0]

# 결과 출력
print(f"전체 태풍 수: {len(df)}")
print(f"영향도가 0이 아닌 태풍 수: {len(df_filtered)}")

# 연도별 영향도가 0이 아닌 태풍 수 계산
yearly_count = df_filtered.groupby('활동년도').size().reset_index(name='태풍 수')

# 그래프 그리기
plt.figure(figsize=(12, 6))
plt.bar(yearly_count['활동년도'], yearly_count['태풍 수'])
plt.title('연도별 영향도가 0이 아닌 태풍 수')
plt.xlabel('연도')
plt.ylabel('태풍 수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 영향도가 0이 아닌 태풍 데이터 저장
df_filtered.to_csv('typhoon_data_filtered.csv', index=False)
print("\n영향도가 0이 아닌 태풍 데이터가 'typhoon_data_filtered.csv' 파일로 저장되었습니다.")

# 영향도가 0이 아닌 태풍 목록 출력
print("\n영향도가 0이 아닌 태풍 목록:")
print(df_filtered[['태풍명', '영문명', '영향도', '활동년도', '활동시기']])