import DataPull
import datetime


def sma(ticker, start, end):
    data = DataPull.get_price_data(ticker, start, end)
    num_days = DataPull.market_days(start, end)
    one_day = datetime.timedelta(1)
    returns = 0
    j = 0
    for i in range(1, num_days+1):
        if DataPull.check_open(start+one_day*i):
            returns += data.loc[DataPull.format_date(start+one_day*i)]['Close'] - data.loc[DataPull.format_date(start+one_day*j)]['Close']
            j = i
    returns = returns / num_days
    return returns


def rate_of_return(ticker, end, n):
    start = end - datetime.timedelta(n)
    data = DataPull.get_price_data(ticker, start, end)
    while not DataPull.check_open(start):
        start += datetime.timedelta(1)
    while not DataPull.check_open(end):
        end -= datetime.timedelta(1)
    returns = data.loc[DataPull.format_date(end)]['Close'] - data.loc[DataPull.format_date(start)]['Close']
    returns = returns / data.loc[DataPull.format_date(start)]['Close']
    print("returns by holding for " + str(n) + " days: " + str(returns))
    return returns

def main():
    print(rate_of_return("AAPL", end=datetime.datetime.today(),n=5))

if __name__ == "__main__":
    main()



