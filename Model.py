import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import math

# this should be rewritten to be able to take specific parameters in the future
def build():
    df = pd.read_csv("stocklist.csv", index_col = 0)
    stocks = df.index.tolist()
    ror = []
    pbr = []
    vol = []
    for stock in stocks:
        data = pd.read_csv(stock+".csv", index_col = 0)
        try:
            prices = data.loc[data.index[-13:-7], 'Last Price']
            currentVol = np.nanmean(data.loc[data.index[-13:-7], 'Volatility 180 D'].values)
            currentROR = rateOfReturn(prices)
            currentPBR = np.nanmean(data.loc[data.index[-13:-7], 'Price to Book'].values)
            if not math.isnan(currentVol) and not math.isnan(currentROR) and  not math.isnan(currentPBR):
                vol.append(currentVol)
                ror.append(currentROR)
                pbr.append(currentPBR)
            else:
                stocks.remove(stock)
        except KeyError:
            stocks.remove(stock)
    ror = np.array(ror)
    vol = np.array(vol)
    pbr = np.array(pbr)
    topTenPercentile = np.nanpercentile(ror, 90)
    isInTopTenPercentile = []
    for i, returns in enumerate(ror):
        if returns > topTenPercentile:
            isInTopTenPercentile.append(1)
        else:
            isInTopTenPercentile.append(-1)

    isInTopTenPercentile = np.array(isInTopTenPercentile)

    from sklearn import tree
    X = np.vstack((ror, pbr, vol)).T
    Y = isInTopTenPercentile
    clf = tree.DecisionTreeClassifier()
    clf.fit(X,Y)

    clfControl = tree.DecisionTreeClassifier()
    clfControl.fit(ror.reshape(-1,1), isInTopTenPercentile)

    stocks = df.index.tolist()
    vol = []
    ror = []
    pbr = []
    for stock in stocks:
        data = pd.read_csv(stock+".csv", index_col = 0)
        try:
            prices = data.loc[data.index[-6:-5], 'Last Price']
            currentVol = np.nanmean(data.loc[data.index[-6:-5], 'Volatility 180 D'].values)
            currentROR = rateOfReturn(prices)
            currentPBR = np.nanmean(data.loc[data.index[-6:-5], 'Price to Book'].values)
            if not math.isnan(currentVol) and not math.isnan(currentROR) and not math.isnan(currentPBR):
                vol.append(currentVol)
                ror.append(currentROR)
                pbr.append(currentPBR)
            else:
                stocks.remove(stock)
        except KeyError:
            stocks.remove(stock)
    ror = np.array(ror)
    vol = np.array(vol)
    pbr = np.array(pbr)
    topTenPercentile = np.nanpercentile(ror, 90)
    isInTopTenPercentile = []
    for i, returns in enumerate(ror):
        if returns > topTenPercentile:
            isInTopTenPercentile.append(1)
        else:
            isInTopTenPercentile.append(-1)

    X = np.vstack((ror, pbr, vol)).T
    predictions = clf.predict(X)
    control = clfControl.predict(ror.reshape(-1,1))

    from sklearn.metrics import accuracy_score
    print(accuracy_score(isInTopTenPercentile, predictions))
    print(accuracy_score(isInTopTenPercentile, control))


    for i, stock in enumerate(stocks):
        if isInTopTenPercentile[i] == 1:
            print(stock)

def decisionTree():
    pass


def rateOfReturn(prices):
    if len(prices)==0:
        return math.nan;
    total = 0
    for i in range(1, len(prices)):
        total += math.log1p(prices.values[i]) - math.log1p(prices.values[i-1])
    return total / len(prices) * 12

if __name__ == "__main__":
    t0 = time.time()
    build()
    print(time.time() - t0, "seconds wait time")
