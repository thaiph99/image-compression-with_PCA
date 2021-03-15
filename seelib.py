from sklearn.decomposition import PCA, IncrementalPCA
import numpy as np

a = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
b = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
c = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
d = [a, b, c]

# print('reshape : ', a.hstack()

# arr = np.concatenate((a, b, c), axis=0)
# print('concatenate : \n', arr)

# arr1 = np.hstack((a, b, c))
# print('hstack : \n', arr1)

# arr2 = np.vstack((a, b, c))
# print('vstack : \n', arr2)

# arr3 = np.dstack((a, b, c))
# print('dstack : \n', arr2)

print(a.shape)

arr4 = np.ones((a.shape[0], a.shape[1], 3), 'int')

arr4[:, :, 0] = a
arr4[:, :, 1] = b
arr4[:, :, 2] = c

print(arr4.shape)
