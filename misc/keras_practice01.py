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
# it takes in the units which is the size of its output
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


#the output of a layer is between 0 and 1 (represents a percentage)


#training data: x is the input data and y is the target data
x_train = np.array([[1, 1, 0],[0.2, 0, 0.9],[0.7, 0.8, 0.41],[0.75, 0.9, 0.3],[0, 0.4, 0.8]])
y_train = np.array([[1],[0],[1],[1],[0]])

#validation data
#x_val = np.array([1, 2, 3])
#y_val = np.array([1, 2, 3])

#create model
model = Sequential()
#model.add(Input(shape=1, name = 'input_layer'))
model.add(Dense(5, input_dim = 3, activation = 'relu')) #elu, relu, linear, sigmoid
model.add(Dense(5, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

#compile model
model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

#train model with fit
model.fit(x_train, y_train, batch_size = 10, epochs = 60, verbose = 1)


_, accuracy = model.evaluate(x_train, y_train)
print('Accuracy: %.2f' % (accuracy*100))


#predict another sample
x_predict = np.array([[0.9,0.8,0.1]])
prediction = model.predict(x_predict)

print('Prediction: %f' % prediction)




# Results:
# loss          optimizer           batch size          epochs          accuracy
# meansqerror   sgd                 5                   20              40, 60x2, 100
# meansqerror   sgd                 10                  40              60x4, 100
# meansqerror   sgd                 10                  60              40x2, 60x2, 80
# meansqerror   adam                10                  60              60x2, 100












