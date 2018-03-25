import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from sklearn import tree


def build(modelType, startIndex, endIndex, target, features):
    df = pd.read_csv("Information Technology.csv", index_col = 0)
    stocks = df.index.tolist()
    allFeatures = []
    allTargets = []
    for stock in stocks:
        featureValues = retrieveData(stock, features, startIndex, endIndex, indexes = [])
        if target == 'Rate of Return':
            targetValues = retrieveData(stock, 'Last Price', endIndex+1, abs(endIndex-startIndex) + endIndex + 1, indexes = [])
        featureValues.dropna(axis = 0, how = 'any', inplace = True)
        targetValues.dropna(axis = 0, how = 'any', inplace = True)
        if featureValues.empty == False and targetValues.empty == False:
            ror = rateOfReturn(targetValues)
            averageValues = featureValues.mean(axis = 0)
            allFeatures.append(averageValues)
            allTargets.append(ror)
    allTargets = getPercentile(allTargets, 90)
    dt = modelType(allTargets, allFeatures)
    visualizeDecisionTreeClassifier(dt, str(startIndex) + ": Top 90")
    return dt


def predict(model, startIndex, endIndex, features):
    df = pd.read_csv("Information Technology.csv", index_col = 0)
    stocks = df.index.tolist()
    allFeatures = []
    addedStocks = []
    for count, stock in enumerate(stocks):
        featureValues = retrieveData(stock, features, startIndex, endIndex, indexes = [])
        featureValues.dropna(axis = 0, how = 'any', inplace = True)
        if featureValues.empty == False:
            averageValues = featureValues.mean(axis = 0)
            allFeatures.append(averageValues)
            addedStocks.append(stock)
    return addedStocks, model.predict(allFeatures)


def printPredictedPerformers(stocks, predictions):
    performers = []
    for i, stock in enumerate(stocks):
        if predictions[i] == 1:
            performers.append(stock)
    print(performers)
    return performers


def test(max, length):
    trees = []
    indexes = np.arange(max * -1 -1 , -1 * length * 3)
    np.random.shuffle(indexes)
    train, validate, test = np.split(indexes, [int(.6*len(indexes)), int(.8*len(indexes))])
    print(train)
    print(validate)
    print(test)
    count = 0
    allPredictions = []
    allActPercentiles = []
    avgPrecision = 0
    precisions = []
    for i in train:
        count += 1
        tree = build(modelType = decisionTreeClassifier, startIndex = i, endIndex = i+length-1, target= 'Rate of Return', features = featureList)
        addedStocks, predictions = predict(tree, startIndex = i+length, endIndex = i+2*length-1, features = featureList)
        actual = []
        for stock in addedStocks:
            actual.append(rateOfReturn(retrieveData(stock, 'Last Price', i+2*length, i+3*length-1, [])))
        actPercentiles = getPercentile(actual, 90)
        allActPercentiles.append(actPercentiles)
        allPredictions.append(predictions)
        if(len(predictions)>0 and len(actPercentiles)>0):
            prec = precision(actPercentiles, predictions)
            avgPrecision = (avgPrecision * (count - 1) + prec)/count
            print("avg: " + str(avgPrecision))
            print(str(count/len(train)*100)+" percent complete")
            precisions.append(prec)
    graphPrecisions(np.asarray(precisions))
    pass

def graphPrecisions(precisionList):
    plt.hist(precisionList, bins='auto')
    plt.show()

# either use startIndex/endIndex or indexes, not both
def retrieveData(ticker, features, startIndex, endIndex, indexes):
    data = pd.read_csv(ticker+".csv", index_col = 0)
    if len(indexes) is not 0:
        values = data.loc[data.index[indexes], features]
    else:
        try:
            values = data.loc[data.index[startIndex:endIndex], features]
        except KeyError:
            return pd.DataFrame({'A' : []})
    return values


def decisionTreeClassifier(targetValues, featureValues):
    X = np.vstack(featureValues)
    if len(featureValues) == 1:
        X = X.reshape(-1,1)
    Y = np.vstack(targetValues)
    clf = tree.DecisionTreeClassifier(max_depth=5,min_samples_leaf=5)
    clf.fit(X,Y)
    return clf


def accuracy(actual, predictions):
    from sklearn.metrics import accuracy_score
    score = accuracy_score(actual, predictions)
    return score


def misclassifications(actual, predictions):
    from sklearn.metrics import zero_one_loss
    score = zero_one_loss(actual, predictions)
    return score


def f1(actual, predictions):
    from sklearn.metrics import f1_score
    score = f1_score(actual, predictions)
    return score


def precision(actual, predictions):
    from sklearn.metrics import precision_score
    score = precision_score(actual, predictions)
    return score

# returns a list of tickers in the top percentile
def getPercentileTickers(tickers, values, percentile):
    if len(tickers)!= len(values):
        raise ValueError('Passed in values and tickers must be the same length')
    cutoff = np.nanpercentile(values, percentile)
    aboveCutoff = []
    for i, val in enumerate(values):
        if val > cutoff:
            aboveCutoff.append(tickers[i])
        else:
            aboveCutoff.append(tickers[i])
    return aboveCutoff


# returns a list of 0, 1 depending upon whether or not its index's ticker
# is in the top percentile
def getPercentile(values, percentile):
    cutoff = np.nanpercentile(values, percentile)
    aboveCutoff = []
    for i, val in enumerate(values):
        if val > cutoff:
            aboveCutoff.append(1)
        else:
            aboveCutoff.append(0)
    return aboveCutoff


# calculates rate of return using logs
def rateOfReturn(prices):
    if len(prices)==0:
        return math.nan;
    if type(prices) == list:
        return (math.log1p(prices[-1]) - math.log1p(prices[0]))
    return (math.log1p(prices.values[-1]) - math.log1p(prices.values[0]))


# exports a graphic representation of a decision tree classifier
def visualizeDecisionTreeClassifier(dtree, name):
    import graphviz
    dot_data = tree.export_graphviz(dtree, out_file=name+'.dot')
    #graph = graphviz.Source(dot_data)
    #graph.render(name)


if __name__ == "__main__":
    t0 = time.time()
    featureList = ['Price to Book', 'Price to Cash Flow', 'Dividend Yield', 'EPS Growth', 'Trailing EPS', 'Total Debt to Total Equity', 'EPS', 'Volatility 180 D', 'Return on Invested Capital', 'Return on Common Equity', 'Return on Assets']
    #test(200,24)
    tree = build(modelType = decisionTreeClassifier, startIndex = -48, endIndex = -25, target= 'Rate of Return', features = featureList)
    addedStocks, predictions = predict(tree, startIndex = -24, endIndex = -1, features = featureList)
    printPredictedPerformers(addedStocks, predictions)
    print(time.time() - t0, "seconds wait time")
