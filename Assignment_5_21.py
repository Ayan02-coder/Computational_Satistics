import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()

X = iris.data
Y = iris.target

X.shape

Y.shape

pca = PCA(n_components=2)

pca.fit(X)

pca.components_

Z = pca.transfrom(X)

Z.shape

plt.scatter(Z[: , 0], Z[ : , 1],c=Y)
