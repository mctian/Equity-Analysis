import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import os
from multiprocessing import Pool


# build a model using a single start and end index
def build(modelType, startIndex, endIndex, target, features, featureLength, targetLength, sector, percentileTarget):
	df = pd.read_csv(sector + ".csv", index_col = 0)
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
		allTargets = allTargets + getPercentile(currentTargets, percentileTarget)
		currentIndex += 1
	dt = modelType(allTargets, allFeatures)
	#visualizeDecisionTreeClassifier(dt, str(startIndex) + ": Top 50")
	return dt


# build a model with a list of start indexes
def buildWithIndexes(modelType, indexes, target, features, featureLength, targetLength, sector, percentileTarget):
	df = pd.read_csv(sector + ".csv", index_col = 0)
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
			allTargets = allTargets + getPercentile(currentTargets, percentileTarget)
	print("Finished data retrieval, starting model training.")
	dt = modelType(allTargets, allFeatures)
	return dt


# build a model with a list of start indexes that also classifiers underperformers
def buildWithIndexesTripleClass(modelType, indexes, target, features, featureLength, targetLength, sector, percentileTarget, percentileAvoid = 0, verbose = False):
	t = time.time()
	df = pd.read_csv(sector + ".csv", index_col = 0)
	stocks = df.index.tolist()
	params = [targetLength, featureLength, stocks, percentileAvoid, percentileTarget, features, target]
	paramList = []
	for i in indexes:
		temp = [i] + (params)
		paramList.append(temp)
	pool = Pool(os.cpu_count())
	temp = pool.map(poolRetrieve, paramList)
	allTargets = []
	allFeatures = []
	for item in temp:
		itemTargets, itemFeatures = zip(*item)
		allTargets.extend(itemTargets)
		allFeatures.extend(itemFeatures)
	print("Finished data retrieval, starting model training. Time taken: " + str(time.time() - t) + " seconds.")
	dt = modelType(allTargets, allFeatures)
	print("Finished fitting. Time taken: " + str(time.time() - t) + " seconds.")
	return dt


# multiprocess function
def poolRetrieve(inputList):
	i = inputList[0]
	targetLength = inputList[1]
	featureLength = inputList[2]
	stocks = inputList[3]
	percentileAvoid = inputList[4]
	percentileTarget = inputList[5]
	features = inputList[6]
	target = inputList[7]
	allFeatures = []
	allTargets = []
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
		allTargets = allTargets + getPercentileTripleClass(currentTargets, percentileTarget, percentileAvoid)
	return zip(allTargets, allFeatures)


# retrieve features for prediction, return added equities and vector of predictions
def predict(model, startIndex, endIndex, features, sector):
	df = pd.read_csv(sector + ".csv", index_col = 0)
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


# retrieve features for prediction, return added equities and vector of predicted probabilities
def predict_probabilities(model, startIndex, endIndex, features, sector):
	df = pd.read_csv(sector + ".csv", index_col = 0)
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
	X = np.vstack(allFeatures)
	#print(X)
	#print(len(X))
	return addedStocks, model.predict_proba(X)


# takes vector of predictions, returns a list of predicted performing stocks
def printPredictedPerformers(stocks, predictions):
	performers = []
	for i, stock in enumerate(stocks):
		if predictions[i] == 1:
			performers.append(stock)
	print(performers)
	return performers


# splits data into test, train, and validation sets
def splitData(max, targetLength, featureLength):
	indexes = np.arange(max * -1 -1 , -1 * (targetLength + featureLength), targetLength)
	np.random.shuffle(indexes)
	train, validate, test = np.split(indexes, [int(.6*len(indexes)), int(.8*len(indexes))])
	print(train)
	print(validate)
	print(test)
	return train, validate, test


# graphs a histogram
def graphPrecisions(list):
	plt.hist(list, bins='auto')
	plt.show()


# retrieves feature data, either use startIndex/endIndex or indexes, not both
def retrieveData(ticker, features, startIndex, endIndex, indexes):
	data = pd.read_csv(ticker+".csv", index_col = 0)
	if len(indexes) is not 0:
		values = data.loc[data.index[indexes], features]
	else:
		try:
			values = data.loc[data.index[len(data) + startIndex:len(data) + endIndex + 1], features]
		except KeyError:
			return pd.DataFrame({'A' : []})
	return values


# sklearn decision tree with set hyperparameters
def decisionTreeClassifier(targetValues, featureValues):
	Y = np.vstack(targetValues)
	Y = Y.reshape(-1,1)
	X = np.vstack(featureValues)
	clf = tree.DecisionTreeClassifier(max_features = 6, max_depth = 10)
	clf.fit(X,Y)
	return clf


# sklearn random forest with set hyperparameters
def randomForestClassifier(targetValues, featureValues):
	Y = np.vstack(targetValues)
	Y = Y.reshape(-1,1)
	X = np.vstack(featureValues)
	clf = RandomForestClassifier(n_estimators = 1000, class_weight = "balanced", \
		min_samples_leaf = 1, n_jobs = -1)
	clf.fit(X,Y.flatten())
	return clf


# sklearn adaboost with set hyperparameters
def adaBoostClassifier(targetValues, featureValues):
	Y = np.vstack(targetValues)
	Y = Y.reshape(-1,1)
	X = np.vstack(featureValues)
	clf = AdaBoostClassifier(n_estimators=10)
	clf.fit(X, Y.flatten())
	return clf


# sklearn accuracy (aka precision and recall)
def accuracy(actual, predictions):
	from sklearn.metrics import accuracy_score
	score = accuracy_score(actual, predictions)
	return score


# sklearn zero one loss
def misclassifications(actual, predictions):
	from sklearn.metrics import zero_one_loss
	score = zero_one_loss(actual, predictions)
	return score


# sklearn f1score
def f1(actual, predictions):
	from sklearn.metrics import f1_score
	score = f1_score(actual, predictions)
	return score


# sklearn precisions
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

# get tickers below a certain nanpercentile
# returns a list of 0, 1 depending upon whether or not its index's ticker
# is below percentile cutoff
def getBelowPercentile(values, percentile):
	cutoff = np.nanpercentile(values, percentile)
	belowCutoff = []
	for i, val in enumerate(values):
		if val < cutoff:
			belowCutoff.append(1)
		else:
			belowCutoff.append(0)
	return belowCutoff


# get tickers below a certain nanpercentile and above a certain nanpercentile
# returns a list of 0, 1, 2
def getPercentileTripleClass(values, abovePercentile, belowPercentile):
	upperCutoff = np.nanpercentile(values, abovePercentile)
	lowerCutoff = np.nanpercentile(values, belowPercentile)
	labels = []
	for i, val in enumerate(values):
		if val < lowerCutoff:
			labels.append(2)
		elif val > upperCutoff:
			labels.append(1)
		else:
			labels.append(0)
	return labels


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
		return (math.log(prices[-1]) - math.log(prices[0]))
	return (math.log(prices.values[-1]) - math.log(prices.values[0]))


# exports a graphic representation of a decision tree classifier
def visualizeDecisionTreeClassifier(dtree, name, featureList):
	import graphviz
	dotfile = open("dtree" + name + ".dot", 'w')
	tree.export_graphviz(dtree, out_file = dotfile, feature_names = featureList)
	dotfile.close()
	return


# approximate dates based on index hardcoded with Ford
# @ index: index of the date, -1 for most recent, 0 for first index
# @ return: datetime object representing the index
def convertIndexToDate(index):
    df = pd.read_csv("F.csv", index_col = 0)
    return df.index[len(df) + index]


def countsBarGraph(counts):
	return

if __name__ == "__main__":
	pass
