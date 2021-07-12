from PIL import Image
import numpy as np 
from matplotlib import pyplot as plt

w, h = 28, 28

file = '../../input/test/test.csv'
data = np.loadtxt(file, skiprows=1, delimiter=',')

row = data[9,:]

arr = np.array(row)
arr = arr.astype('int32')

mat = arr.reshape(h, w)

plt.gray()
fig = plt.imshow(mat)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig('result.png')
plt.show()

