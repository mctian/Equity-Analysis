import datetime
import Model
import numpy as np

def rebalanceIndexes(startIndex, endIndex):
    return list(range(startIndex, endIndex, 3))

# ToDo
def convertIndexToDate(index):
    pass
    return


def getPortfolio(rebalanceIndex, max, targetLength, featureLength, featureList, probabilityThreshold):
    indexes = np.arange(max * -1 -1 * rebalanceIndex, -1 * (targetLength + featureLength), targetLength)
    tree = Model.buildWithIndexes(modelType = Model.randomForestClassifier, indexes = indexes, \
        target= 'Rate of Return', features = featureList, featureLength = 12, targetLength = 3)
    print ("Labels: ")
    print (tree.classes_)
    print ("Importances: ")
    print (tree.feature_importances_)
    addedStocks, probabilities = Model.predict_probabilities(tree,
        startIndex = rebalanceIndex - 11, endIndex = rebalanceIndex, features = featureList)
    stockTuples = zip(addedStocks, probabilities)
    stockTuples = list(filter(lambda x: x[1][1] > probabilityThreshold, stockTuples))
    print (zip(*stockTuples)[0])
    return zip(*stockTuples)[0]

def getReturns(portfolio, startIndex, length):
    returns = 0
    print(portfolio)
    print(startIndex)
    for stock in portfolio:
        returns += 1 / len(portfolio) * rateOfReturn(retrieveData(
            stock, 'Last Price', startIndex, startIndex + length -1, []))
    return returns

if __name__ == "__main__":
    featureList = ['Price to Book', 'Dividend Yield', 'Total Debt to Total Equity',
        'Return on Invested Capital', 'Return on Common Equity']
    port = getPortfolio(12,200,3,12,featureList,0.6)
    print(port)
    #portfolios = list(map(lambda x: getPortfolio(x,200,3,12,featureList,0.6), rebalanceIndexes(-12,-1)))
    #portTups = zip(portfolios, rebalanceIndexes(-12,-1))
    #for portTup in portTups:
    #    print(getReturns(portTup[0], portTup[1], 3))
