import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
df = pd.read_csv('rainfall_daily_avg.csv')

# Convert '날짜' column to datetime format
df['날짜'] = pd.to_datetime(df['날짜'], errors='coerce')

# Plot the total rainfall over time
plt.figure(figsize=(12, 6))
plt.plot(df['날짜'], df['total_rainfall'], label='Total Rainfall')
plt.xlabel('Date')
plt.ylabel('Total Rainfall (mm)')
plt.title('Daily Average Rainfall Over Time')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()