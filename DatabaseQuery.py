from pymongo import MongoClient
import pandas as pd
import time


if __name__ == '__main__':
    client = MongoClient('localhost', 27017)
    db = client.stocklists
    db2 = client.stocks
    collection = db['Financials']
    cursor = collection.find()
    result = pd.DataFrame(list(cursor))
    t = time.time()
    for ticker in result['Tickers']:
        collection = db2[ticker]
        cursor = collection.find()
        stock = pd.DataFrame(list(cursor))
        print(stock.sort_values(by=['Date']))
    print(time.time() - t)
    t = time.time()

