from sklearn.decomposition import PCA
import numpy as np
import scipy.misc                  # for loading image
from matplotlib.pyplot import imread
np.random.seed(1)

# filename structure
path = 'YALE/unpadded/'  # path to the database
ids = range(1, 16)  # 15 persons
states = ['centerlight', 'glasses', 'happy', 'leftlight',
          'noglasses', 'normal', 'rightlight', 'sad',
          'sleepy', 'surprised', 'wink']
prefix = 'subject'
surfix = '.pgm'
# data dimension
h, w, K = 116, 98, 100  # hight, weight, new dim
D = h * w
N = len(states)*15
# collect all data
X = np.zeros((D, N))
cnt = 0
for person_id in range(1, 16):
    for state in states:
        fn = path + prefix + str(person_id).zfill(2) + '.' + state + surfix
        print(fn)
        X[:, cnt] = imread(fn).reshape(D)
        # misc.imread
        cnt += 1

# Doing PCA, note that each row is a datapoint
pca = PCA(n_components=K)  # K = 100
pca.fit(X.T)
# projection matrix
U = pca.components_.T
