import yfinance as yf
import csv


stock_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
fields = ["Stock Name", "PE Ratio", "Dividened Yields", "Market Cap", "Rolling Average"]

desired_pe_ratio = 20  
desired_dividend_yield = 0.02
desired_market_cap = 10
desired_moving_average = 0

stock_data = []

def binary_convert(values):
    returning = []
    print(values)
    for value in values:
        returning.append(str(value))

    return returning

for stock_symbol in stock_symbols:

    stock = yf.Ticker(stock_symbol)

    historical_data = stock.history(period="1mo")

    stock_info = stock.info

    pe_ratio = stock.info['trailingPE']
    volume = stock.info['volume']
    market_cap = stock.info['marketCap']

    print(str(pe_ratio))

    #moving_average_50 = historical_data['Close'].rolling(window=50).mean()

    binary_stock_data = binary_convert([stock_symbol,pe_ratio,volume,market_cap,1])
    stock_data.append(binary_stock_data)
    print(stock_data)

try:
    with open("stock_data.csv", 'w') as file:
        csvwriter = csv.writer(file)
        csvwriter.writerow(fields)
        csvwriter.writerows(stock_data)

finally:
    file.close()
    