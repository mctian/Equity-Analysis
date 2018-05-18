import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from sklearn import tree


def build(modelType, startIndex, endIndex, target, features, featureLength, targetLength):
    df = pd.read_csv("Financials.csv", index_col = 0)
    stocks = df.index.tolist()
    allFeatures = []
    allTargets = []
    currentIndex = startIndex
    while currentIndex + featureLength + targetLength - 1 != endIndex + 1:
        currentTargets = []
        for stock in stocks:
            featureValues = retrieveData(stock, features, currentIndex, currentIndex + featureLength - 1, indexes = [])
            if target == 'Rate of Return':
                targetValues = retrieveData(stock, 'Last Price', currentIndex + featureLength, currentIndex + featureLength + targetLength - 1, indexes = [])
            featureValues.dropna(axis = 0, how = 'any', inplace = True)
            targetValues.dropna(axis = 0, how = 'any', inplace = True)
            if featureValues.empty == False and targetValues.empty == False:
                ror = rateOfReturn(targetValues)
                averageValues = featureValues.mean(axis = 0)
                allFeatures.append(averageValues)
                currentTargets.append(ror)
        allTargets = allTargets + getPercentile(currentTargets, 50)
        currentIndex += 1
        print(len(allTargets))
        print(len(allFeatures))
    dt = modelType(allTargets, allFeatures)
    visualizeDecisionTreeClassifier(dt, str(startIndex) + ": Top 50")
    return dt


def buildWithIndexes(modelType, indexes, target, features, featureLength, targetLength):
    df = pd.read_csv("Financials.csv", index_col = 0)
    stocks = df.index.tolist()
    allFeatures = []
    allTargets = []
    for i in indexes:
        for j in range(i,i+targetLength):
            currentTargets = []
            for stock in stocks:
                featureValues = retrieveData(stock, features, j, j + featureLength - 1, indexes = [])
                if target == 'Rate of Return':
                    targetValues = retrieveData(stock, 'Last Price', j + featureLength, j + featureLength + targetLength - 1, indexes = [])
                featureValues.dropna(axis = 0, how = 'any', inplace = True)
                targetValues.dropna(axis = 0, how = 'any', inplace = True)
                if featureValues.empty == False and targetValues.empty == False:
                    ror = rateOfReturn(targetValues)
                    averageValues = featureValues.mean(axis = 0)
                    allFeatures.append(averageValues)
                    currentTargets.append(ror)
            allTargets = allTargets + getPercentile(currentTargets, 50)
            print(len(allTargets))
            print(len(allFeatures))
    dt = modelType(allTargets, allFeatures)
    return dt


def predict(model, startIndex, endIndex, features):
    df = pd.read_csv("Financials.csv", index_col = 0)
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


def predict_probabilities(model, startIndex, endIndex, features):
    df = pd.read_csv("Financials.csv", index_col = 0)
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
    return addedStocks, model.predict_proba(allFeatures)


def printPredictedPerformers(stocks, predictions):
    performers = []
    for i, stock in enumerate(stocks):
        if predictions[i] == 1:
            performers.append(stock)
    print(performers)
    return performers


def splitData(max, targetLength, featureLength):
    trees = []
    indexes = np.arange(max * -1 -1 , -1 * (targetLength + featureLength), targetLength)
    np.random.shuffle(indexes)
    train, validate, test = np.split(indexes, [int(.6*len(indexes)), int(.8*len(indexes))])
    print(train)
    print(validate)
    print(test)
    return train, validate, test

def graphPrecisions(precisionList, name):
    plt.hist(precisionList, bins='auto')
    plt.title(name)
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
    Y = np.vstack(targetValues)
    Y = Y.reshape(-1,1)
    X = np.vstack(featureValues)
    clf = tree.DecisionTreeClassifier(max_features = 6, min_samples_leaf = 5)
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
    if len(prices) == 0:
        return math.nan;
    if type(prices) == list:
        return (math.log1p(prices[-1]) - math.log1p(prices[0]))
    return (math.log1p(prices.values[-1]) - math.log1p(prices.values[0]))


# exports a graphic representation of a decision tree classifier
def visualizeDecisionTreeClassifier(dtree, name):
    #import graphviz
    #dot_data = tree.export_graphviz(dtree, out_file=name+'.dot')
    #graph = graphviz.Source(dot_data)
    #graph.render(name)
    pass


if __name__ == "__main__":
    t0 = time.time()
    featureList = ['Price to Book', 'Price to Cash Flow', 'Dividend Yield', 'EPS Growth', 'Trailing EPS', 'Total Debt to Total Equity', 'EPS', 'Volatility 180 D', 'Return on Invested Capital', 'Return on Common Equity', 'Return on Assets']
    train, validate, test = splitData(350,3,36)
    tree = buildWithIndexes(modelType = decisionTreeClassifier, indexes = train, target= 'Rate of Return', features = featureList, featureLength = 36, targetLength = 3)
    above50 = {'90':0, '80':0, '70':0}
    seen = {'90':0, '80':0, '70':0}
    for i in validate:
        #addedStocks, predictions = predict(tree, startIndex = i, endIndex = i+35, features = featureList)
        #print(predictions)
        addedStocks, probabilities = predict_probabilities(tree, startIndex = i, endIndex = i+35, features = featureList)
        actual = []
        better90 = []
        better80 = []
        better70 = []
        for stock in addedStocks:
            actual.append(rateOfReturn(retrieveData(stock, 'Last Price', i+36, i+38, [])))
        for i in range(len(probabilities)):
            if probabilities[i][0] > 0.7:
                better70.append(1)
            else:
                better70.append(0)
            if probabilities[i][0] > 0.8:
                better80.append(1)
            else:
                better80.append(0)
            if probabilities[i][0] > 0.9:
                better90.append(1)
            else:
                better90.append(0)
        if len(better90) != 0:
            above50['90'] = (precision(getPercentile(actual, 50), better90) * len(better90) + above50['90'] * seen['90']) /  (seen['90'] + len(better90))
            seen['90'] = seen['90'] + len(better90)
        if len(better80) != 0:
            above50['80'] = (precision(getPercentile(actual, 50), better80) * len(better80) + above50['80'] * seen['80']) /  (seen['80'] + len(better80))
            seen['80'] = seen['80'] + len(better80)
        if len(better70) != 0:
            above50['70'] = (precision(getPercentile(actual, 50), better70) * len(better70) + above50['70'] * seen['70']) /  (seen['70'] + len(better70))
            seen['70'] = seen['70'] + len(better70)
    print(time.time() - t0, "seconds wait time")
    print('90: ' + str(above50['90']))
    print('80: ' + str(above50['80']))
    print('70: ' + str(above50['70']))
