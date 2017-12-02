import DataPull
import datetime
import time
import scipy


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
    return returns

def main():
    start_time = time.time()
    tickers = DataPull.get_all_tickers(market="Russell3000")
    valid = []
    returns = []
    for ticker in tickers:
        try:
            returns.append(rate_of_return(ticker, end=datetime.datetime.today(), n=30))
            valid.append(ticker)
        except:
            pass
    threshold = scipy.percentile(returns, 90)
    portfolio = []
    for i in range(0, len(returns)):
        if returns[i] > threshold:
            portfolio.append(valid[i])
    print(portfolio)
    print("Duration of program: " + str(time.time()-start_time))


if __name__ == "__main__":
    main()



