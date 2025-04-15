import csv
import pandas as pd

df = pd.read_csv("stock_data.csv")


desired_pe_ratio = 30 
desired_dividend_yield = 4000000
desired_market_cap = 3000000000000
desired_moving_average = 1

average_pe_ratio = 20 
average_dividend_yield = 3000000
average_market_cap = 2000000000000
average_moving_average = 1

poor_pe_ratio = 10
poor_dividend_yield = 1000000
poor_market_cap = 1000000000000
poor_market_average = 0

def convert(desired,average,poor, trueValue):
    if trueValue > desired:
        return "11"
    elif trueValue > average:
        return "10"
    elif trueValue > poor:
        return "01"
    return "00"

binary_values = [bin(i)[2:] for i in range(len(df))]
binary_values = [value.zfill(len(binary_values[-1])) for value in binary_values]

df.index = binary_values

df["PE Ratio"] = df["PE Ratio"].map(lambda x: convert(desired_pe_ratio,average_pe_ratio,poor_pe_ratio,x))
df["Dividened Yields"] = df["Dividened Yields"].map(lambda x: convert(desired_dividend_yield,average_dividend_yield,poor_dividend_yield,x))
df["Market Cap"] = df["Market Cap"].map(lambda x: convert(desired_market_cap,average_market_cap,poor_market_cap,x))

df.drop(columns=["Stock Name"])

print(df)
        


