import numpy as np
from PIL import Image 
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import model_from_json

img_rows, img_cols = 28, 28

#load json and create model
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

#load weights into model
model.load_weights("model.h5")
print()
print("Model successfully loaded from disk")

#compile model
model.compile(loss = keras.losses.categorical_crossentropy,
            optimizer = 'adam',
            metrics = ['accuracy'])

#resise target
target = img_rows

print()
print("Make sure that you inserted the image in the 'test' folder")
test_image = input("Enter the image file's name : ")

img = Image.open('../input/test/'+test_image)
width = img.size[0]
height = img.size[1]

#vertical image resize
if width <= height:
	wpercent = (target/float(width))
	hsize = int((float(height)*float(wpercent)))
	img = img.resize((target,hsize), Image.ANTIALIAS)
	left, top, right, bottom = 0, ((hsize//2) - (target//2)), target, ((hsize//2) + (target//2))
	img = img.crop((left, top, right, bottom))

#horizontal image resize
else:
	hpercent = (target/float(height))
	wsize = int((float(width)*float(hpercent)))
	img = img.resize((wsize,target), Image.ANTIALIAS)
	left, top, right, bottom = ((wsize//2) - (target//2)), 0, ((wsize//2) + (target//2)), target
	img = img.crop((left, top, right, bottom))

#image to array IMP
array = np.array(img)
#if r,g,b
try:
    if (len(array[0][0] == 3)):
        #axis = 2 takes the mean of r, g and b
        array = np.mean(img, axis=2) 
        array = array
except:
    #grayscale image
    array = 255 - 255*array

array.astype('int32')

#turn all to white background
white = 0
black = 0
for j in range(28):
    for i in [0, 1, 2, 25, 26, 27]:
        if (array[i][j] < 128):
            black += 1
        else:
            white += 1
for i in range(3, 25):
    for j in [0, 1, 2, 25, 26, 27]:
        if (array[i][j] < 128):
            black += 1
        else:
            white += 1
#turn white bg to black bg
if white > black:
   for i in range(len(array)):
    for j in range(len(array[0])):
        array[i][j] = 255 - array[i][j]

#1D array
arr = array.flatten()

#clean image
#120+ is pure white
#99- is pure black
#white goes from 100 to 255
i = 0
while i < len(arr):
    if arr[i] >= 120:
        arr[i] = 255
    elif arr[i] >= 100:
        arr[i] += 135
    else:
        arr[i] = 0
    i += 1

#1 -> 1 image
#1 -> grayscale
test_arr = arr.reshape(1, img_rows, img_cols, 1)

#normalisation
test_arr = test_arr/255

#pred is a 1 x num_classes matrix
pred = model.predict(test_arr)
print()
print("The recognized digit is : ",np.argmax(pred))


#visualize image as the machine sees it
mat = test_arr.reshape(img_rows, img_cols)

plt.gray()
fig = plt.imshow(mat)
fig.axes.get_xaxis().set_visible(False)
fig.axes.get_yaxis().set_visible(False)
plt.savefig('processed_image.png')
plt.show()

#NOTES:
#THE TESTED IMAGE SHOULD HAVE A UNIFORM BACKGROUND
#IF THE IMAGE HAS A BLACK FRAME WITH A WHITE BACKGROUND, THE IMAGE CLEANING WON'T WORK
#TRY TO CENTER THE DIGIT IN THE IMAGE AS MUCH AS POSSIBLE 