from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

A = imread('images/test.jpg')
X = np.mean(A, -1);  # Convert RGB to grayscale

img = plt.imshow(X)
img.set_cmap('gray')
# plt.show()

U, S, VT = np.linalg.svd(X, full_matrices=False)
S = np.diag(S)

j = 0
for r in (5, 20, 100):
    Xapprox = U[:, :r] @ S[0:r, :r] @ VT[:r, :]
    plt.imshow(Xapprox)
    plt.title('r = ' + str(r))
    plt.show()
    j += 1