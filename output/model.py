#Validation precision: 98.7%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import model_from_json

img_rows, img_cols = 28, 28
num_classes = 10
val_split = 0.2

def preprocess(data):
    y = data[:,0]
    X = data[:,1:]
    num_images = data.shape[0]

    #20% is given to validation data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = val_split, random_state = 2)

    #encode 0->9 to hot vectors(binary values)
    out_y_train = keras.utils.to_categorical(y_train, num_classes)
    out_y_val = keras.utils.to_categorical(y_val, num_classes)
    
    #grayscale so channel = 1
    out_X_train = X_train.reshape(int((1 - val_split)*num_images), img_rows, img_cols, 1)
    out_X_val = X_val.reshape(int(val_split*num_images), img_rows, img_cols, 1)

    #normalisation
    out_X_train = out_X_train/255
    out_X_val = out_X_val/255

    return out_X_train, out_X_val, out_y_train, out_y_val


#data of white digits on black background
file = "../input/train/train.csv"
data = np.loadtxt(file, skiprows=1, delimiter=',')
X_train, X_val, y_train, y_val = preprocess(data)

#define model
model = Sequential()

#first layer
#32 convolutions
#relu is the rectifier, adds non-linearity to the model
model.add(Conv2D(32,
                kernel_size = (5, 5),
                activation = 'relu',
                input_shape = (img_rows, img_cols, 1)))

#hidden layers
model.add(Conv2D(32,
                kernel_size = (5, 5),
                activation = 'relu'))
#takes max of two successive pixels
model.add(MaxPool2D(pool_size=(2,2)))
#removes parts to prevent overfiting
model.add(Dropout(0.25))
model.add(Conv2D(64,
                kernel_size = (3, 3),
                activation = 'relu'))
model.add(Conv2D(64,
                kernel_size = (3, 3),
                activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.25))
#convert output of previous layers into a 1D layer
model.add(Flatten())
model.add(Dense(256,
                activation = 'relu'))
model.add(Dropout(0.5))

#prediction layer
#softmax turns the last layer output into a vector of probabilities
model.add(Dense(10,
                activation = 'softmax'))

#compile model
#optimizer adam to automatically choose searching parameter for back-propagation and gradient descend
model.compile(loss = keras.losses.categorical_crossentropy,
            optimizer = 'adam',
            metrics = ['accuracy'])

#set a learning rate annealer
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

#avoid running excessive epochs
earlystopper = EarlyStopping(monitor='val_loss', min_delta=0,
                             patience=1, verbose=1, mode='auto')
#data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

#fit model
#batch size: how many images to process at a time
#epochs: how many times to repeat the procedure
history = model.fit(datagen.flow(X_train,y_train, batch_size = 100),
                              epochs = 10, 
                              validation_data = (X_val,y_val),
                              #verbose = 2, #if we want compact input
                              steps_per_epoch=X_train.shape[0] // 100,
                              callbacks=[learning_rate_reduction, earlystopper])


#SAVE
#serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
#serialize weights to HDF5
model.save_weights("model.h5")
print("Model successfully saved to disk")



#PLOT LOSS AND ACCURACY CURVES FOR TRAINING AND VALIDATION
#NB: validation accuracy should always be greater than training accuracy to make sure that the model doesn't overfit
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.show()


#SHOW CONFUISION MATRIX
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Predict the values from the validation dataset
y_pred = model.predict(X_val)
# Convert predictions classes to one hot vectors 
y_pred_classes = np.argmax(y_pred,axis = 1) 
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1) 
# compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes) 
# plot the confusion matrix
plot_confusion_matrix(confusion_mtx, classes = range(10)) 


#SHOW TOP 6 ERRORS
# Errors are difference between predicted labels and true labels
errors = (y_pred_classes - y_true != 0)

y_pred_classes_errors = y_pred_classes[errors]
y_pred_errors = y_pred[errors]
y_true_errors = y_true[errors]
X_val_errors = X_val[errors]

def display_errors(errors_index,img_errors,pred_errors, obs_errors):
    """ This function shows 6 images with their predicted and real labels"""
    n = 0
    nrows = 2
    ncols = 3
    fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True)
    for row in range(nrows):
        for col in range(ncols):
            error = errors_index[n]
            ax[row,col].imshow((img_errors[error]).reshape((28,28)))
            ax[row,col].set_title("Predicted label :{}\nTrue label :{}".format(pred_errors[error],obs_errors[error]))
            n += 1
    plt.show()

# Probabilities of the wrong predicted numbers
y_pred_errors_prob = np.max(y_pred_errors,axis = 1)

# Predicted probabilities of the true values in the error set
true_prob_errors = np.diagonal(np.take(y_pred_errors, y_true_errors, axis=1))

# Difference between the probability of the predicted label and the true label
delta_pred_true_errors = y_pred_errors_prob - true_prob_errors

# Sorted list of the delta prob errors
sorted_dela_errors = np.argsort(delta_pred_true_errors)

# Top 6 errors 
most_important_errors = sorted_dela_errors[-6:]

# Show the top 6 errors
display_errors(most_important_errors, X_val_errors, y_pred_classes_errors, y_true_errors)
