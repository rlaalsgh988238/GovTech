import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr
import numpy as np

class RainfallData:
    def __init__(self, date, rainfall):
        self.date = date
        self.rainfall = float(rainfall)

    def __repr__(self):
        return f"RainfallData(date={self.date}, rainfall={self.rainfall})"
class ApplePriceData:
    def __init__(self, date, item, grade, unit, min_price, max_price, avg_price, day_change, prev_day_avg, prev_week_avg, prev_year_avg):
        self.date = date
        self.item = item
        self.grade = grade
        self.unit = unit
        self.min_price = int(min_price.replace(",", ""))
        self.max_price = int(max_price.replace(",", ""))
        self.avg_price = int(avg_price.replace(",", ""))
        self.day_change = int(day_change.replace(",", ""))
        self.prev_day_avg = self.parse_comparison(prev_day_avg)
        self.prev_week_avg = self.parse_comparison(prev_week_avg)
        self.prev_year_avg = self.parse_comparison(prev_year_avg)

    def parse_comparison(self, value):
        try:
            number, _ = value.split()
            return int(number.replace(",", ""))
        except ValueError:
            # If splitting doesn't work, return a default or handle the error
            return 0

    def __repr__(self):
        return f"ApplePriceData(date={self.date}, item={self.item}, grade={self.grade}, unit={self.unit}, avg_price={self.avg_price})"

def parse_apple_price_data_from_file(file_path):
    apple_price_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            if line and not line.startswith("날짜") and not line.startswith("1.본 자료는"):
                parts = line.split("\t")
                if len(parts) >= 11:
                    apple_data = ApplePriceData(
                        date=parts[0],
                        item=parts[1],
                        grade=parts[2],
                        unit=parts[3],
                        min_price=parts[4],
                        max_price=parts[5],
                        avg_price=parts[6],
                        day_change=parts[7],
                        prev_day_avg=parts[8],
                        prev_week_avg=parts[9],
                        prev_year_avg=parts[10]
                    )
                    apple_price_list.append(apple_data)
    return apple_price_list
def getAppleList(item,grade,dataList):
    AppleList = []
    for data in dataList:
        if(data.item == item and data.grade == grade):
            AppleList.append(data)
    AppleList = sort_apple_list_by_date(AppleList)
    return AppleList
def plot_apple_prices(data_list, title):
    # 데이터프레임으로 변환
    df = pd.DataFrame([data.__dict__ for data in data_list])

    # 날짜를 datetime 형식으로 변환
    df['date'] = pd.to_datetime(df['date'])

    # 그래프 설정
    plt.figure(figsize=(10, 6))
    plt.plot(df['date'], df['avg_price'], marker='o', linestyle='-')
    # 그래프 제목 및 라벨 설정
    plt.title(title)
    plt.xlabel('date')
    plt.ylabel('avg_price (won)')
    plt.xticks(rotation=45)
    plt.grid(True)

    # 그래프 출력
    plt.tight_layout()
    plt.show()
def sort_apple_list_by_date(apple_list):
    """주어진 사과 데이터 리스트를 날짜에 따라 정렬합니다."""
    return sorted(apple_list, key=lambda x: datetime.strptime(x.date, '%Y.%m.%d'))
def parse_rainfall_data_from_file(file_path):
    rainfall_data_list = []
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip()
            # 데이터가 있는 줄만 처리
            if line and not line.startswith("[") and not line.startswith("자료구분") and not line.startswith("날짜"):
                parts = line.split(",")
                if len(parts) == 3:
                    date, _, rainfall = parts
                    rainfall_data = RainfallData(date=date, rainfall=rainfall)
                    rainfall_data_list.append(rainfall_data)
    return rainfall_data_list

def analyze_correlation(rainfall_data_list, apple_price_data_list, title):
    # 날짜별로 데이터를 매칭하기 위해 딕셔너리 생성
    rainfall_dict = {data.date: data.rainfall for data in rainfall_data_list}

    # ApplePriceData의 날짜 형식을 YYYY-MM-DD로 변환
    apple_price_dict = {datetime.strptime(data.date, "%Y.%m.%d").strftime("%Y-%m-%d"): data.avg_price for data in apple_price_data_list}

    # 공통 날짜에 대한 리스트 생성
    common_dates = set(rainfall_dict.keys()).intersection(set(apple_price_dict.keys()))

    # 공통 날짜에 해당하는 데이터만 추출
    rainfall_values = [rainfall_dict[date] for date in common_dates]
    apple_prices = [apple_price_dict[date] for date in common_dates]

    # 데이터가 충분히 있는지 확인
    if len(rainfall_values) < 2 or len(apple_prices) < 2:
        print("공통 날짜가 충분하지 않습니다. 상관관계를 계산하려면 최소 2개의 공통 날짜가 필요합니다.")
        return

    # 상관계수와 p-value 계산
    correlation, p_value = pearsonr(rainfall_values, apple_prices)
    print(f"correlation: {correlation:.2f}, p-value: {p_value:.4f}")

    # 그래프 출력
    plt.figure(figsize=(10, 6))
    plt.scatter(rainfall_values, apple_prices)
    plt.title(title)
    plt.xlabel('RainFall (mm)')
    plt.ylabel('Price (won)')
    plt.grid(True)

    # 상관계수와 p-value를 그래프에 표시
    plt.text(0.05, 0.95, f'correlation: {correlation:.2f}\np-value: {p_value:.4f}',
             transform=plt.gca().transAxes, verticalalignment='top', fontsize=10,
             bbox=dict(facecolor='white', alpha=0.5))

    plt.show()
def makeAppleTitle(item, grade):
    item_dict = {
        '사과 부사': 'busa apple',
        '사과 홍로': 'hongro apple',
        '사과 로얄부사': 'royal busa apple',
        '사과 미시마': 'mishima apple',
        '사과 썸머킹': 'summerking apple'
    }

    grade_dict = {
        '특': 'extra',
        '상': 'large',
        '보통': 'medium',
        '하': 'small'
    }

    # 아이템과 등급을 영어로 변환하여 제목 생성
    item_english = item_dict.get(item, 'unknown item')
    grade_english = grade_dict.get(grade, 'unknown grade')

    # 제목 생성
    title = f"{item_english} {grade_english} avg_price change"
    return title
def plot_variance(rainfall_data_list, apple_price_data_list, title):
    # 날짜별로 데이터를 매칭하기 위해 딕셔너리 생성
    rainfall_dict = {data.date: data.rainfall for data in rainfall_data_list}

    # ApplePriceData의 날짜 형식을 YYYY-MM-DD로 변환
    apple_price_dict = {datetime.strptime(data.date, "%Y.%m.%d").strftime("%Y-%m-%d"): data.avg_price for data in apple_price_data_list}

    # 공통 날짜에 대한 리스트 생성
    common_dates = set(rainfall_dict.keys()).intersection(set(apple_price_dict.keys()))

    # 공통 날짜에 해당하는 데이터만 추출
    rainfall_values = [rainfall_dict[date] for date in common_dates]
    apple_prices = [apple_price_dict[date] for date in common_dates]

    # 데이터가 충분히 있는지 확인
    if len(rainfall_values) < 2 or len(apple_prices) < 2:
        print("공통 날짜가 충분하지 않습니다. 분산을 계산하려면 최소 2개의 공통 날짜가 필요합니다.")
        return

    # 분산 계산
    rainfall_variance = np.var(rainfall_values)
    apple_price_variance = np.var(apple_prices)

    # 분산을 그래프로 출력
    plt.figure(figsize=(10, 6))
    plt.bar(['Rainfall', 'Apple Price'], [rainfall_variance, apple_price_variance], color=['blue', 'green'])
    plt.title(title)
    plt.ylabel('Variance')
    plt.grid(axis='y')

    # 분산 값을 그래프에 표시
    plt.text(0, rainfall_variance, f'{rainfall_variance:.2f}', ha='center', va='bottom', fontsize=10, color='blue')
    plt.text(1, apple_price_variance, f'{apple_price_variance:.2f}', ha='center', va='bottom', fontsize=10, color='green')

    plt.show()

filePathList = [
    'AppleData/20240724-0731.txt',
    'AppleData/20240716-0723.txt',
    'AppleData/20240708-0715.txt',
    'AppleData/20240701-0707.txt'
]

item = '사과 썸머킹'
grade = '특'
rainDataList = parse_rainfall_data_from_file('../RainData/2020_2024JulyRainfallAll.txt')
title = makeAppleTitle(item, grade)



dataCount = 0
appleDataList = []



for file_path in filePathList:
    apple_price_list = parse_apple_price_data_from_file(file_path)
    for data in apple_price_list:
        dataCount = dataCount+1
        appleDataList.append(data)

print(dataCount)
print(rainDataList)

appleList = getAppleList(item, grade, appleDataList)
print(appleList)

analyze_correlation(rainDataList, appleList, title)