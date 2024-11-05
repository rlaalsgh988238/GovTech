import csv

# 입력 파일명과 출력 파일명 설정
input_file = '2000_2024_Apple_Honglo_Data.txt'
output_file = '2000_2024_Apple_Honglo_Data.csv'

# 입력 파일 읽기 및 CSV 파일로 쓰기
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
    # CSV writer 객체 생성
    csv_writer = csv.writer(outfile)

    # 헤더 쓰기
    csv_writer.writerow(['날짜', '중량(kg)', '등급', '가격(원)'])

    # 첫 줄 (헤더) 건너뛰기
    next(infile)

    # 각 줄을 읽어 CSV로 쓰기
    for line in infile:
        # 줄바꿈 문자 제거 및 쉼표로 분리
        row = line.strip().split(',')
        if len(row) == 4:  # 유효한 데이터 행인지 확인
            csv_writer.writerow(row)

print(f"변환 완료: {output_file}")
