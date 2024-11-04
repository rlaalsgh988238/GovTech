import csv
from datetime import datetime

# 입력 파일과 출력 파일 이름 설정
input_file = '2000_2024_OilData.txt'
output_file = 'output.csv'

# 결과를 저장할 리스트
result = []

# 입력 파일 읽기
with open(input_file, 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    next(reader)  # 헤더 건너뛰기

    for row in reader:
        date = row[0].strip()
        opening_price = float(row[2].replace(',', ''))
        exchange_rate = float(row[8].replace(',', ''))

        # 한화 계산
        korean_won = opening_price * exchange_rate

        # 날짜 형식 변경 (YYYY-MM-DD)
        date_obj = datetime.strptime(date, '%Y- %m- %d')
        formatted_date = date_obj.strftime('%Y-%m-%d')

        result.append([formatted_date, korean_won])

# 결과를 CSV 파일로 저장
with open(output_file, 'w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['날짜', '한화'])  # 헤더 쓰기
    writer.writerows(result)

print(f"처리가 완료되었습니다. 결과가 {output_file}에 저장되었습니다.")
