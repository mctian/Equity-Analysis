import DataPull
import datetime

def sma(ticker, start, end):
    data = DataPull.get_price_data(ticker, start, end)
    num_days = (end - start).days
    one_day = datetime.timedelta(1)
    returns = 0
    for i in range(0, num_days):
        returns += data.loc[DataPull.format_date(start+one_day*(i+1))]['Close'] - data.loc[DataPull.format_date(start+one_day*(i))]['Close']
    returns = returns / num_days
    return returns

print(sma('AAPL', start=datetime.datetime(2017, 1, 4, 0, 0, 0, 0), end=datetime.datetime(2017, 1, 18, 0, 0, 0, 0))) # year, month, day

