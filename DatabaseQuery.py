import pymongo
import pandas as pd
import time
import collections
import numpy as np
import json
import itertools


def getValidFields():
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    fields = db['Fields'].find(projection = {'_id': False})
    print(pd.DataFrame(list(fields))['Field'].values)
    return fields


# Do NOT RUN IF 'Fields' ALREADY EXISTS
def initValidFields():
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db2 = client.stocks
    fieldCollection = db['Fields']
    fields = pd.DataFrame(list(db2['A'].find(projection = {'_id':False}, sort = [('Date', -1)], limit = 1))).columns.values
    for field in fields:
        fieldDoc = {'Field': field}
        fieldCollection.insert_one(fieldDoc)
    return


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


def getDataAnom(start, end = '2018-01-31', sector = 'stocklist', fields = 'All'):
    if fields == 'All':
        fields = ['Date', 'EPS Growth', 'Volatility 180 D', 'Trailing EPS', 'Price to Cash Flow', 'EPS', 'Volume', 'Return on Assets', 'Price to Book', 'Dividend Yield', 'Total Debt to Total Equity', 'Return on Invested Capital', 'Return on Common Equity']
        proj = {}
        proj["_id"] = False
        for field in fields:
            proj[field] = True
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db2 = client.stocks
    stocks = pd.DataFrame(list(db[sector].find()))['Tickers'].values
    
    dataDict = {}
    for field in fields:
        dataDict[field] = np.array([], dtype = np.float64)

    for stock in stocks:
        df = pd.DataFrame(list(db2[stock].find(filter = {'$and':[{'Date': {'$gte':start}}, {'Date': {'$lte':end}}]}, projection = proj)))
        for feature in fields:
            try:
                if feature != 'Date':
                    arr = np.array(df[feature].values, dtype = np.float64)
                else:
                    arr = np.array(df[feature].values)
                dataDict[feature] = np.append(dataDict[feature], arr)
            except KeyError:
                try:
                    arr = np.array(list(itertools.repeat(np.nan, df.shape[0])))
                    dataDict[feature] = np.append(dataDict[feature], arr)
                except:
                    pass
            except ValueError:
                pass

    return dataDict


def getDataAnomQuarterly(start, end = '2018-01-31', sector = 'stocklist', fields = 'All'):
    if fields == 'All':
        fields = ['Date', 'EPS Growth', 'Volatility 180 D', 'Trailing EPS', 'Price to Cash Flow', 'EPS', 'Volume', 'Return on Assets', 'Price to Book', 'Dividend Yield', 'Total Debt to Total Equity', 'Return on Invested Capital', 'Return on Common Equity']
        proj = dict()
        proj["_id"] = False
        for field in fields:
            proj[field] = True
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db2 = client.stocks
    stocks = pd.DataFrame(list(db[sector].find()))['Tickers'].values
    
    dataDict = {}
    for field in fields:
        dataDict[field] = np.array([], dtype = np.float64)
    for stock in stocks:
        df = pd.DataFrame(list(db2[stock].find(filter = {'$and':[{'Date': {'$gte':start}}, {'Date': {'$lte':end}}]}, projection = proj)))
        numrows = df.shape[0] - df.shape[0] % 3
        if numrows == 0:
            continue
        for feature in fields:
            try:
                if feature != 'Date':
                    arr = np.array(df[feature].values, dtype = np.float64)[0:numrows].reshape(3,-1)
                    arr = np.nanmean(arr, axis = 0)
                else:
                    arr = np.array(df[feature].values)[0:numrows:3]
                dataDict[feature] = np.append(dataDict[feature], arr)
            except KeyError:
                arr = np.array(list(itertools.repeat(np.nan, int(numrows/3))))
                dataDict[feature] = np.append(dataDict[feature], arr)
            except ValueError:
                print('Warning: ValueError')
                pass
    return dataDict


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
            pass
    return


if __name__ == '__main__':
    getValidFields()
