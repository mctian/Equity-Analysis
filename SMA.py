import DataPull


def sma(ticker, start, end):
    data = DataPull.get_price_data(ticker, start, end)
    avg = sum(data['Close']) / len(data['Close'])
    return avg


def main():
    print(sma('AAPL', start = DataPull.format_date(1,1,2008), end = DataPull.format_date(2,2,2009)))

if __name__ == "__main__":
    main()
