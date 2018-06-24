import pymongo
import pandas as pd
import time
import collections
import numpy as np
import json
import itertools


def getValidDateIndexes():
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    indexes = db['Dates'].index_information()
    print(indexes)
    return indexes


# DO NOT RUN IF 'Dates' ALREADY EXISTS
def initValidDates():
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db2 = client.stocks
    collection = db2['F'] 
    cursor = collection.find(projection={'_id':False, 'Date':True}, sort = [('Date',-1)])
    dateCollection = db['Dates']
    for date in pd.DataFrame(list(cursor))['Date']:
        dateDoc = {'Date': date}
        print(dateDoc)
        dateCollection.insert_one(dateDoc)
    return


def deleteDuplicates():
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db2 = client.stocks
    stocks = pd.DataFrame(list(db['stocklist'].find()))['Tickers'].values
    for stock in stocks:
        data = pd.DataFrame(list(db2[stock].find(projection = {"_id":False})))
        data.drop_duplicates(inplace=True)
        db2[stock].drop()
        db2[stock].insert_many(json.loads(data.T.to_json()).values())
    return


def initDateIndexes():
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db2 = client.stocks
    stocks = pd.DataFrame(list(db['stocklist'].find()))['Tickers'].values
    for stock in stocks:
        db2[stock].create_index([('Date', -1)], unique = True)
    return


def initDateIndexesForDateList():
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db['Dates'].create_index([('Date', -1)], unique = True)
    return


def getData(start, end = '2018-01-31', sector = 'stocklist', fields = 'All'):
    if fields == 'All':
        fields = ['EPS Growth', 'Volatility 180 D', 'Trailing EPS', 'Price to Cash Flow', 'EPS', 'Volume', 'Return on Assets', 'Price to Book', 'Dividend Yield', 'Total Debt to Total Equity', 'Return on Invested Capital', 'Return on Common Equity']
        proj = fields
        #proj = list(zip(fields, itertools.repeat(True)))
        #print(proj)
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db2 = client.stocks
    stocks = pd.DataFrame(list(db[sector].find()))['Tickers'].values
    datalist = []
    for stock in stocks:
        data = db2[stock].find(filter = {'$and':[{'Date': {'$gte':start}}, {'Date': {'$lte':end}}]}, projection = proj)
        datalist.append(pd.DataFrame(list(data)))
    print(datalist)
    return datalist



def test():
    featureList = ['EPS Growth', 'Volatility 180 D', 'Trailing EPS', 'Price to Cash Flow', 'EPS', 'Volume', 'Return on Assets', 'Price to Book', 'Dividend Yield', 'Total Debt to Total Equity', 'Return on Invested Capital', 'Return on Common Equity']
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db2 = client.stocks
    collection = db['stocklist']
    cursor = collection.find()
    result = pd.DataFrame(list(cursor))
    t = time.time()
    epsgrowth = []
    count = 0
    for ticker in result['Tickers']:
        collection = db2[ticker]
        cursor = collection.find(filter = {"Date":{"$gt":"2017-12-1"}})
        stockdata = pd.DataFrame(list(cursor))
        if count == 6:
            print(stockdata)
            print(cursor.explain())
            count += 1
        else:
            count += 1
        for feature in featureList:
            try:
                stockfeature = stockdata[feature].dropna().values
                if feature == "EPS Growth":
                    epsgrowth.extend(stockfeature)
            except:
                pass
    return
 

if __name__ == '__main__':
    getData('2017-1-1', sector = 'TelecommunicationServices')
    
