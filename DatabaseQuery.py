import pymongo
import pandas as pd
import time
import collections
import numpy as np
import json
import itertools


def getValidFields(includePrice = True):
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    fields = db['Fields'].find(projection = {'_id': False})
    fields = list(pd.DataFrame(list(fields))['Field'].values)
    if not includePrice:
        fields.remove('Last Price')
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


def getDataAnom(start, end = '2018-01-31', sector = 'stocklist', fields = 'All', includePrice = False, periodLimit = 0):
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db2 = client.stocks
    stocks = pd.DataFrame(list(db[sector].find()))['Tickers'].values

    if fields == 'All':
        fields = getValidFields(includePrice)

    proj = {} 
    proj["_id"] = False
    for field in fields:
        proj[field] = True
    
    dataDict = {}

    total = 0

    for field in fields:
        dataDict[field] = np.array([], dtype = np.float32)

        for stock in stocks:
            df = pd.DataFrame(list(db2[stock].find(filter = {'$and':[{'Date': {'$gte':start}}, {'Date': {'$lte':end}}]}, projection = proj)))
            if periodLimit > 0:
                numrows = df.shape[0] - df.shape[0] % periodLimit 
            else:
                numrows = df.shape[0]
        
            total += numrows
        
            if numrows == 0:
                continue
            for feature in fields:
                try:
                    if feature != 'Date':
                        arr = np.array(df[feature].values, dtype = np.float32)[0:numrows]
                    else:
                        arr = np.array(df[feature].values)[0:numrows]
                    dataDict[feature] = np.append(dataDict[feature], arr)
                except KeyError:
                    arr = np.array(list(itertools.repeat(np.nan, numrows)))
                    dataDict[feature] = np.append(dataDict[feature], arr)
                except ValueError:
                    print('Warning: ValueError')
                    pass

    print(total)

    return dataDict


def getDataAnomQuarterly(start, end = '2018-01-31', sector = 'stocklist', fields = 'All', includePrice = False):
    client = pymongo.MongoClient('localhost', 27017, maxPoolSize=100)
    db = client.stocklists
    db2 = client.stocks
    stocks = pd.DataFrame(list(db[sector].find()))['Tickers'].values

    if fields == 'All':
        fields = getValidFields(includePrice)
    
    proj = {}
    proj["_id"] = False
    for field in fields:
        proj[field] = True
    
    dataDict = {}

    for field in fields:
        dataDict[field] = np.array([], dtype = np.float32)

    total = 0

    for stock in stocks:
        df = pd.DataFrame(list(db2[stock].find(filter = {'$and':[{'Date': {'$gte':start}}, {'Date': {'$lte':end}}]}, projection = proj)))
        numrows = df.shape[0] - df.shape[0] % 3
        
        total += numrows
        
        if numrows == 0:
            continue
        for feature in fields:
            try:
                if feature != 'Date':
                    arr = np.array(df[feature].values, dtype = np.float32)[0:numrows].reshape(3,-1)
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

    print(total)

    return dataDict


def consolidateDataDict(dataDict):
    arr = np.array([])
    keys = list(dataDict)

    try:
        keys.remove('Date')
    except ValueError:
        pass

    for key in keys:
        arr = np.append(arr, dataDict[key])
    arr = arr.reshape(-1, len(keys))
    return arr


def calculateRateOfReturn(prices, period = 3):
    prices = prices.flatten()
    length = prices.shape[0] - prices.shape[0] % period 
    startPrices = prices[0:length:period]
    endPrices = prices[period - 1:length:period]
    return np.log(endPrices) - np.log(startPrices)


# Note: does not match functionality, currently it replaces NaN with 0
def dropNan(x, y):
    dfx = pd.DataFrame(x)
    dfy = pd.DataFrame(y.T)
    df = pd.concat([dfx, dfy], axis = 1)
    df.dropna(axis = 0, how = 'any', inplace = True)
    print(df)
    return


def test():
    t = time.time()
    dataDict = getDataAnomQuarterly('2016-1-1')
    arr = consolidateDataDict(dataDict)
    print(arr.shape)
    print(time.time() - t)

    t = time.time()
    priceDict = getDataAnom('2016-1-1', fields = ['Last Price'], periodLimit = 3)
    priceArr = consolidateDataDict(priceDict)
    rorArr = calculateRateOfReturn(priceArr)
    print(rorArr.shape)
    print(time.time() - t)

    t = time.time()
    from sklearn.ensemble import RandomForestClassifier

    Y = rorArr
    X = arr

    dropNan(X,Y)

    rfc = RandomForestClassifier(n_estimators = 1000, class_weight = "balanced", min_samples_leaf = 1, n_jobs = -1)
    rfc.fit(X,Y)
    
    print(time.time() - t)

    return


if __name__ == '__main__':
    test()
   
