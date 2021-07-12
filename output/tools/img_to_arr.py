from PIL import Image
import numpy as np 


img = Image.open('result.png')
#2D array
array = np.array(img)
#if r,g,b
try:
	if (len(array[0][0] == 3)):
		#axis = 2 takes the mean of r, g and b
		array = np.mean(img, axis=2) 
		print("RGB")
except:
	array = 255 - 255*array
	print("Grayscale")
#1D array
arr = array.flatten()
arr = arr.astype('int32')

#clean image
i = 0
for px in arr:
	if px <= 127:
		if arr[i]-70 < 0:
			arr[i] = 0
		else:
			arr[i] -= 70
	else:
		if arr[i]+70 > 255:
			arr[i] = 255
		else:
			arr[i] += 70
	i += 1

############
#it becomes#
############

mat = arr.reshape(28, 28)
from matplotlib import pyplot as plt
plt.gray()
fig = plt.imshow(mat)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.show()
