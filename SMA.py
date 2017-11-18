import DataPull
import datetime


def sma(ticker, end, n):
    start = end - datetime.timedelta(n)
    data = DataPull.get_price_data(ticker, start, end)
    avg = 0
    for i in range(0, n):
        if DataPull.check_open(start+datetime.timedelta(i)):
            avg += data.loc[DataPull.format_date(start+datetime.timedelta(i))]['Close']
    avg = avg / DataPull.market_days(start, end)
    return avg


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
    print(sma("AAPL", end=datetime.datetime.today(), n=5))
    print(rate_of_return("AAPL", end=datetime.datetime.today(),n=5))

if __name__ == "__main__":
    main()



