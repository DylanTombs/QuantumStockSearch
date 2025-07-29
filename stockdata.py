import os
import pandas as pd
import numpy as np
import glob

# Path to your folder containing CSV files
dataFolder = "Data/"
summaryData = []

# List all CSVs in the folder
csvFiles = glob.glob(os.path.join(dataFolder, "*.csv"))

# For each stock CSV, extract stats
for file in csvFiles:
    stockName = os.path.basename(file).replace(".csv", "")
    df = pd.read_csv(file, parse_dates=["date"])

    df.sort_values("date", inplace=True)
    df.dropna(inplace=True)

    # Calculate daily returns
    df["return"] = df["close"].pct_change()
    df["volatility"] = df["return"].rolling(window=20).std()

    # 50-day and 200-day moving averages
    df["ma_50"] = df["close"].rolling(window=50).mean()
    df["ma_200"] = df["close"].rolling(window=200).mean()

    # Most recent values (for features)
    latest = df.iloc[-1]
    meanReturn = df["return"].mean()
    stdReturn = df["return"].std()
    latestVolatility = latest["volatility"]
    ma50 = latest["ma_50"]
    ma200 = latest["ma_200"]

    summaryData.append([
        stockName,
        round(meanReturn, 4),
        round(stdReturn, 4),
        round(latestVolatility, 4),
        round(ma50, 2),
        round(ma200, 2)
    ])

# Save results
summaryDf = pd.DataFrame(summaryData, columns=[
    "Stock", "MeanReturn", "StdReturn", "Volatility", "MA50", "MA200"
])

summaryDf.to_csv("processed_stock_summary.csv", index=False)
print("✅ Stock data processed and saved to 'processed_stock_summary.csv'")
