import pandas as pd

# Load the CSV files
apple_df = pd.read_csv('../apple_honglo/apple_하.csv')
noyear_df = pd.read_csv('noyear_without_price.csv')

# Ensure the date format is consistent for merging
apple_df['날짜'] = pd.to_datetime(apple_df['날짜'], errors='coerce')
noyear_df['날짜'] = pd.to_datetime(noyear_df['날짜'], errors='coerce')

# Merge the dataframes on '날짜'
merged_df = pd.merge(noyear_df, apple_df[['날짜', '가격(원)']], on='날짜', how='left')

# Sort by date to calculate the previous day's price
merged_df = merged_df.sort_values(by='날짜')

# Add the previous day's price column
merged_df['이전_가격'] = merged_df['가격(원)'].shift(1)

# Save the merged dataframe to a new CSV file
merged_df.to_csv('merged_apple하_noyear.csv', index=False)