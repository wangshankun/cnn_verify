from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataSet = []
fileIn = open('./test.txt')
for line in fileIn.readlines():
        lineArr = line.strip().split()
        dataSet.append([float(lineArr[0]), float(lineArr[1])])

X=np.array(dataSet)
model  = KMeans(n_clusters=3, random_state=9)
model.fit(X)
#print(pd.Series(model.labels_).value_counts())
#print(pd.DataFrame(model.cluster_centers_))
#print(model.cluster_centers_)
#print(np.argmax(np.bincount(model.labels_)))
print(model.cluster_centers_[np.argmax(np.bincount(model.labels_))])
plt.scatter(X[:, 0], X[:, 1], c=(model.labels_))
plt.show()
