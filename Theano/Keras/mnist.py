#%%     #creates separate cell to execute

from keras.datasets import mnist #import MNIST dataset
from keras.models import Sequential #import the type of model
from keras.layers.core import Dense, Dropout, Activation, Flatten #import layers
from keras.layers.convolutional import Convolution2D, MaxPooling2D #import convolution layers
from keras.utils import np_utils

import matplotlib #to plot import matplotlib
import matplotlib.pyplot as plt

#%%    #in this cell we define our parameters

batch_size = 128  #batch size to train
nb_classes = 10   #number of output classes
nb_epoch = 12     #number of epochs to train

img_rows, img_cols = 28, 28  #input image dimensions
nb_filters = 32              #number of convolutional filters to use in each layer
nb_pool = 2                  #size of pooling area for max pooling, 2x2
nb_conv = 3                  #convolution kernel size, 3x3

#%%
# the data, shuffle and split between train and test sets
# X_train and X_test are pixels, y_train and y_test are levels from 0 to 9
# you can see what is inside by typing X_test.shape and y_test shape commands
# X_train are 60000 pictures, 1 channel, 28x28 pixels
# y_train are 60000 labels for them
# X_test are 10000 pictires, 1 channel, 28x28 pixels
# y_test are 10000 labels for them 
(X_train, y_train), (X_test, y_test) = mnist.load_data() 

# reshape the data
# X_train.shape[0] - number of samples,
# 1 - channel, img_rows - image rows, img_cols - image columns
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

#convert X_train and X_test data to 'float32' format
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#then we normalize the data, dividing it by 255, the highest intensity
X_train /= 255
X_test /= 255

# Print the shape of training and testing data
print('X_train shape:', X_train.shape)

# Print how many samples you have
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices,
# for example from "2" to "[0 0 1 0 0 0 0 0 0 0]
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#now let's plot X_train example #4606
i = 4606
plt.imshow(X_train[i, 0], interpolation = 'nearest')
print("label :", Y_train[i,:])

#%% in this cell we define a model of neural network

model = Sequential() # we will use sequential type of model 

# we add first layer to neural network, type of the layer is Convolution 2D,
# with 32 convolutional filters, kernel size 3x3
model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode = 'valid',
                        input_shape = (1, img_rows, img_cols))) # number of channels 1, 28x28 pixels

# now we add activation function to convolutional neurons of the 1st layer,
# it will be "Rectified Linear Unit" function, RELU
convout1 = Activation('relu')
# we add activation function to model to visualize the data
model.add(convout1)

# we add second convolutional layer
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
# and add to 2nd layer the activation function
convout2 = Activation('relu')
model.add(convout2) # add this one to visualize data later 

# we add 3rd layer, type maxpooling, pooling area 2x2
model.add(MaxPooling2D(pool_size = (nb_pool, nb_pool)))

# we add 4th layer, type dropout, which works as a regularizer
model.add(Dropout(0.25))

model.add(Flatten())

#we add 5th layer, consisting of 128 neurons
model.add(Dense(128))
model.add(Activation('relu')) # activation function for them is RELU as well

#add 6th dropout layer
model.add(Dropout(0.5))

#last 7th layer will consist of 10 neurons, same as number of classes for output
model.add(Dense(nb_classes))
model.add(Activation('softmax')) # for last layer we use SoftMax activation function

# and define optimizer and a loss function for a model
model.compile(optimizer='adadelta', loss='categorical_crossentropy',
              metrics=['accuracy'])

#%% in this cell we will train the neural network model

model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch,
          show_accuracy=True, verbose=1, validation_data = (X_test, Y_test))

model.fit(X_train, Y_train, batch_size = batch_size, nb_epoch = nb_epoch,
          show_accuracy = True, verbose = 1, validation_split = 0.2)
                       
#%% in this cell we evaluate the model we trained
                       
score = model.evaluate(X_test, Y_test, show_accuracy = True, verbose = 0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# here we will predict what is six elements of data we have in X_test
# to see how neural network model works and recognize the numbers 
print(model.predict_classes(X_test[1:7]))

#and let's see what labels these images have
print(Y_test[1:7])

# this neural network model can recognize 20000 numbers in 1 minute, using CPU  















