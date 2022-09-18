import keras as keras #machine learning library
from keras.layers import Input,Dense,Lambda, concatenate, add
from keras.models import Model, Sequential
from keras import optimizers
import numpy as np #for math functions
import jax.numpy as jnp


# keras models:
# sequential vs functional (original code uses functional)
# sequential: add one layer at a time
# functional: for more advanced models (inputs/outputs can share layers?)

# "activation" is the function to be done in that node (the activation funciton)
# each layer is a set of nodes
# shape or input_shape is needed for the layer to know what the input looks like
# (like dimensions of a vector?)

# Dense is a type of basic layer in keras
# it takes in the number of nodes to be in that layer
# the output of the previous layer is always automatically assigned to be the input
# for the next layer (unless it is the first, where you have to specify input shape)

# Loss:
# to compute the error between the expected and actual output
# a loss function is required to compile the model

# Optimizers are used to choose better weight values for the nodes
# keras has built-in optimizer functions

# A fit function is used to train the model (? not quite sure what this means)

# steps:
# create training data and "validation data" (correct output?)
# create a model and add layers
# compile the model
# apply the fit() function


#training data: x is the input data and y is the target data
x_train = np.array([1, 2, 3])
y_train = np.array([1, 2, 3])

#validation data
x_val = np.array([1, 2, 3])
y_val = np.array([1, 2, 3])

#create model
model = Sequential()
model.add(Input(shape=1, name = 'input_layer'))
model.add(Dense(1, activation = 'linear')) #elu, relu, linear

#compile model
model.compile(loss = 'mean_squared_error', optimizer = 'sgd', metrics = ['accuracy'])

#train model with fit
model.fit(x_train, y_train, batch_size = 1, epochs = 10, verbose = 1)


accuracy = model.evaluate(x_train, y_train)

print(accuracy*100)





