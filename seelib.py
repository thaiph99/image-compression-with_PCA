from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np

a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
b = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
c = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
d = [a, b, c]

ipca = IncrementalPCA(n_components=200)
