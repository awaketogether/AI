import pandas as pd
import seaborn as sns; sns.set_theme()
from sklearn import preprocessing
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib
from scipy.cluster.hierarchy import *
from sklearn.cluster import AgglomerativeClustering
import sys

#PRE PROCESS

dataset = pd.read_csv(sys.argv[1])
dataset.drop(['state'], axis=1, inplace=True)

plotSingleNight = 1

if plotSingleNight == 1:
    dataset = dataset.head(601)

X = dataset[['axis1', 'axis2', 'axis3']]

scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)

dataset['axis1'] = X_scaled[:, 0]
dataset['axis2'] = X_scaled[:, 1]
dataset['axis3'] = X_scaled[:, 2]
dataset['meanMov'] = [((obj[0] + obj[1] + obj[2])/3) for obj in X_scaled]

array2d = dataset['meanMov'].values.reshape(-1, 1)

scaler = preprocessing.StandardScaler().fit(array2d)
X_scaled = scaler.transform(array2d)

#### PROCESS

fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(15, 10), dpi=80)

if plotSingleNight == 1:

    dates = dataset['timestamp']

    dates = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]

    x = matplotlib.dates.date2num(dates)
    formatter = matplotlib.dates.DateFormatter('%H:%M:%S')

    ax1.xaxis.set_major_formatter(formatter)
    plt.setp(ax1.get_xticklabels(), rotation = 15)

    ax1.scatter(
        x, dataset['meanMov'],
        c='white', marker='o',
        edgecolor='black', s=50
    )

clusterAnalysis = 0
if clusterAnalysis == 1:
    dendrogram = dendrogram(linkage(X_scaled, method='ward'))
    plt.show()

model = AgglomerativeClustering(n_clusters=None, distance_threshold=1.5, affinity='euclidean', linkage='complete')
model.fit_predict(X_scaled)
labels = model.labels_

dates = dataset['timestamp']
dates = [datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in dates]

x_dates = matplotlib.dates.date2num(dates)
formatter = matplotlib.dates.DateFormatter('%H:%M:%S')

ax2.xaxis.set_major_formatter(formatter)
plt.setp(ax2.get_xticklabels(), rotation=15)

ax2.scatter(
    x_dates, X_scaled,
    marker='o',
    s=10,
    c=model.labels_, cmap='rainbow'
)

plt.show()
