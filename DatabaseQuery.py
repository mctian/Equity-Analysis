from pymongo import MongoClient
import pandas as pd
import time


if __name__ == '__main__':
    user = input("Enter database username: ")
    password = input("Enter database password for " + user + ": ")
    ticker =  input("Enter ticker: ")
    client = MongoClient('localhost', 27017)
    db = client.stocklists
    db2 = client.stocks
    collection = db['stocklist']
    cursor = collection.find()
    result = pd.DataFrame(list(cursor))
    t = time.time()
    for ticker in result['Tickers']:
        collection = db[ticker]
        cursor = collection.find()
        stock = pd.DataFrame(list(cursor))
    print(time.time() - t)
    t = time.time()

    tickers = pd.read_csv('stocklist.csv')['Tickers']
    for ticker in tickers:
        a = pd.read_csv(ticker+".csv")
    print(time.time() - t)
