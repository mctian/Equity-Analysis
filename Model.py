import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math


def build(modelType, startIndex, endIndex, target, features):
    df = pd.read_csv("stocklist.csv", index_col = 0)
    stocks = df.index.tolist()
    for stock in stocks:
        targetValues = retrieveData(stock, startIndex, endIndex, target)
        featureValues = retrieveData(stock, startIndex, endIndex, features)
        print(featureValues)
    return


def test():
    return


def retrieveData(ticker, startIndex, endIndex, features):
    data = pd.read_csv(ticker+".csv", index_col = 0)
    values = data.loc[data.index[startIndex:endIndex], features]
    return values


def decisionTreeClassifier(targetValues, featureValues):
    from sklearn import tree
    X = np.vstack(featureValues).T
    if len(features) == 1:
        X = X.reshape(-1,1)
    Y = targetValues
    clf = tree.DecisionTreeClassifier()
    clf.fit(X,Y)
    return clf


def accuracy(actual, predictions):
    from sklearn.metrics import accuracy_score
    scores = []
    for prediction in predictions:
        score = accuracy_score(actual, prediction)
        scores.append(score)
    print(scores)
    return scores


# returns a list of tickers in the top percentile
def getPercentileTickers(tickers, values, percentile):
    if len(tickers)!= len(values):
        raise ValueError('Passed in values and tickers must be the same length')
    cutoff = np.nanpercentile(values, percentiles)
    aboveCutoff = []
    for i, val in enumerate(values):
        if val > cutoff:
            aboveCutoff.append(tickers[i])
        else:
            aboveCutoff.append(tickers[i])
    return aboveCutoff


# returns a list of -1, 1 depending upon whether or not its index's ticker
# is in the top percentile
def getPercentile(values, percentile):
    cutoff = np.nanpercentile(values, percentiles)
    aboveCutoff = []
    for i, val in enumerate(values):
        if val > cutoff:
            aboveCutoff.append(1)
        else:
            aboveCutoff.append(-1)
    return aboveCutoff


def rateOfReturn(prices):
    if len(prices)==0:
        return math.nan;
    return (math.log1p(prices.values[0]) - math.log1p(prices.values[-1])) / len(prices) * 12


if __name__ == "__main__":
    t0 = time.time()
    featureList = ['Price to Book', 'Price to Cash Flow']
    build(modelType = decisionTreeClassifier, startIndex = -7, endIndex = -1, target= 'Last Price', features = featureList)
    print(time.time() - t0, "seconds wait time")
