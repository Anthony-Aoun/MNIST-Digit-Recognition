import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import model_from_json

img_rows, img_cols = 28, 28

def preprocess_test_data(test_data):
    X = test_data[:,:]
    num_images = test_data.shape[0]
    #grayscale so channel = 1
    out_X = X.reshape(num_images, img_rows, img_cols, 1)
    #normalisation
    out_X = out_X/255
    return out_X

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

#det data
test_file = '../input/test/test.csv'
test_data = np.loadtxt(test_file, skiprows=1, delimiter=',')

test_X = preprocess_test_data(test_data)

#preds is a num_examples x num_classes matrix
preds = model.predict(test_X)

#export results as csv
output = pd.DataFrame({'nb predicted' : [np.argmax(pred) for pred in preds]})
output.to_csv('predictions.csv', index = False)
print()
print("Results saved as 'predictions.csv'")