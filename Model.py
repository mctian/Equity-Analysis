import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


def build(modelType, startIndex, endIndex, target, features, featureLength, targetLength):
    df = pd.read_csv("Financials.csv", index_col = 0)
    stocks = df.index.tolist()
    allFeatures = []
    allTargets = []
    currentIndex = startIndex
    while currentIndex + featureLength + targetLength - 1 != endIndex + 1:
        currentTargets = []
        for stock in stocks:
            featureValues = retrieveData(stock, features, currentIndex, \
                currentIndex + featureLength - 1, indexes = [])
            if target == 'Rate of Return':
                targetValues = retrieveData(stock, 'Last Price', currentIndex + \
                    featureLength, currentIndex + featureLength + targetLength - 1, indexes = [])
            featureValues.dropna(axis = 0, how = 'any', inplace = True)
            targetValues.dropna(axis = 0, how = 'any', inplace = True)
            if featureValues.empty == False and targetValues.empty == False:
                ror = rateOfReturn(targetValues)
                averageValues = featureValues.mean(axis = 0)
                allFeatures.append(averageValues)
                currentTargets.append(ror)
        allTargets = allTargets + getPercentile(currentTargets, 90)
        currentIndex += 1
    dt = modelType(allTargets, allFeatures)
    #visualizeDecisionTreeClassifier(dt, str(startIndex) + ": Top 50")
    return dt


def buildWithIndexes(modelType, indexes, target, features, featureLength, targetLength):
    df = pd.read_csv("Financials.csv", index_col = 0)
    stocks = df.index.tolist()
    allFeatures = []
    allTargets = []
    count = 0
    for i in indexes:
        print("Index: " + str(i))
        print(str(count/len(indexes)*100) + " percent complete with preparing data.")
        count += 1
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
            allTargets = allTargets + getPercentile(currentTargets, 90)
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
    clf = tree.DecisionTreeClassifier(max_features = 6, max_depth = 10)
    clf.fit(X,Y)
    return clf


def randomForestClassifier(targetValues, featureValues):
    Y = np.vstack(targetValues)
    Y = Y.reshape(-1,1)
    X = np.vstack(featureValues)
    clf = RandomForestClassifier(n_estimators=120, max_depth=6, class_weight = "balanced", \
        min_samples_leaf=4)
    clf.fit(X,Y.flatten())
    return clf


def adaBoostClassifier(targetValues, featureValues):
    Y = np.vstack(targetValues)
    Y = Y.reshape(-1,1)
    X = np.vstack(featureValues)
    clf = AdaBoostClassifier(n_estimators=10)
    clf.fit(X, Y.flatten())
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
    score = precision_score(actual, predictions, pos_label = 1, average='binary')
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
    import graphviz
    dot_data = tree.export_graphviz(dtree, out_file=name+'.dot')
    graph = graphviz.Source(dot_data)
    graph.render(name)
    pass


if __name__ == "__main__":
    t0 = time.time()
    featureList = ['Price to Book', 'Dividend Yield', 'Total Debt to Total Equity',
        'Return on Invested Capital', 'Return on Common Equity']
    train, validate, test = splitData(200,3,12)
    tree = buildWithIndexes(modelType = randomForestClassifier, indexes = train, \
        target= 'Rate of Return', features = featureList, featureLength = 12, targetLength = 3)
    print("Labels: ")
    print(tree.classes_)
    print("Importances: ")
    print(tree.feature_importances_)
    #print("OOB Scores: ")
    #print(tree.oob_score_)
    precisions = {'seen':{},'above75':{}, 'above50':{}, 'above25':{}, 'above90':{}}
    betterThan = {}
    for prob in range(0,100,5):
        precisions['seen'][prob] = 0
        precisions['above75'][prob] = 0
        precisions['above50'][prob] = 0
        precisions['above25'][prob] = 0
        precisions['above90'][prob] = 0
    for i in validate:
        addedStocks, probabilities = predict_probabilities(tree,
            startIndex = i, endIndex = i+11, features = featureList)
        actual = []
        for prob in range(0,100,5):
            betterThan[prob] = []
        for stock in addedStocks:
            actual.append(rateOfReturn(retrieveData(
                stock, 'Last Price', i+12, i+14, [])))
        for i in range(len(probabilities)):
            for prob in range(0,100,5):
                if probabilities[i][1] > prob / 100:
                    betterThan[prob].append(1)
                else:
                    betterThan[prob].append(0)
        for prob in range(0,100,5):
            if sum(betterThan[prob]) > 0:
                precisions['above75'][prob] = (precision(getPercentile(actual, 75),
                    betterThan[prob]) * len(betterThan[prob]) + precisions['above75'][prob] \
                    * precisions['seen'][prob]) / (precisions['seen'][prob] + len(betterThan[prob]))
                precisions['above50'][prob] = (precision(getPercentile(actual, 50),
                    betterThan[prob]) * len(betterThan[prob]) + precisions['above50'][prob] \
                    * precisions['seen'][prob]) / (precisions['seen'][prob] + len(betterThan[prob]))
                precisions['above25'][prob] = (precision(getPercentile(actual, 25),
                    betterThan[prob]) * len(betterThan[prob]) + precisions['above25'][prob] \
                    * precisions['seen'][prob]) / (precisions['seen'][prob] + len(betterThan[prob]))
                precisions['above90'][prob] = (precision(getPercentile(actual, 90),
                    betterThan[prob]) * len(betterThan[prob]) + precisions['above90'][prob] \
                    * precisions['seen'][prob]) / (precisions['seen'][prob] + len(betterThan[prob]))
                precisions['seen'][prob] = precisions['seen'][prob] + len(betterThan[prob])

    x90 = []
    y90 = []
    x75 = []
    x50 = []
    x25 = []
    y75 = []
    y50 = []
    y25 = []
    for prob in range(0,100,5):
        print(str(prob) + " above 90th percentile: " + str(precisions['above90'][prob]))
        print(str(prob) + " above 75th percentile: " + str(precisions['above75'][prob]))
        print(str(prob) + " above 50th percentile: " + str(precisions['above50'][prob]))
        print(str(prob) + " above 25th percentile: " + str(precisions['above25'][prob]))
        if (precisions['above90'][prob]) > 0:
            x90.append(prob)
            y90.append(precisions['above90'][prob])
        if (precisions['above75'][prob]) > 0:
            x75.append(prob)
            y75.append(precisions['above75'][prob])
        if (precisions['above50'][prob]) > 0:
            x50.append(prob)
            y50.append(precisions['above50'][prob])
        if (precisions['above25'][prob]) > 0:
            x25.append(prob)
            y25.append(precisions['above25'][prob])

    print(time.time() - t0, "seconds wait time")
    fig = plt.figure()
    axes = plt.gca()
    axes.set_xlim([0,100])
    axes.invert_xaxis()
    plt.title("Precisions")
    plt.xlabel('Predicted probability of being in class 90th percentile')
    plt.ylabel('Precision for the percentile of each line')
    plt.plot(x25, y25, color = 'g', label = "25th percentile")
    plt.plot(x50, y50, color = 'b', label = "50th percentile")
    plt.plot(x75, y75, color = 'r', label = "75th percentile")
    plt.plot(x90, y90, color = 'k', label = "90th percentile")
    plt.legend()
    fig.savefig(str(time.time()) + 'test.jpg')
    plt.show()
