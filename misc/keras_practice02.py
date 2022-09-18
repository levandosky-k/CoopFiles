#Example keras program from
#https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
import keras as keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.metrics import Precision
from keras import optimizers
import numpy as np
from matplotlib import pyplot
from random import randrange


#important numbers
train_size = 400
batch_size = 20
epochs = 200
predict_size = 700 - train_size


#custom metric functions
#y_true is the actual y value given by the training example
#y_pred is the value output by the model

#absolute value of difference between y_true and y_pred
def difference(y_true, y_pred):
    return abs(y_true - y_pred)

#returns true value of y
def true(y_true, y_pred):
    return y_true

#returns predicted value of y
def pred(y_true, y_pred):
    return y_pred






#training data
#X is a matrix where each row represents an example of a patient,
#and each column represents a feature of the patient
#y is a vector of 0s and 1s for each patient. A 1 represents
#having diabetes, and a 0 represents no diabetes

dataset = np.loadtxt('pima-indians-diabetes.csv', delimiter=',')
X = dataset[0:train_size,0:8]
y = dataset[0:train_size,8]


#create model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


#compile model
model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
              metrics = ['accuracy'])


#fit model
history = model.fit(X, y, batch_size = 20, epochs = 100)


#print accuracy
#_, accuracy = model.evaluate(X, y)
#print('Accuracy: %.2f' % (accuracy*100))



#data to use for predictions
size = 700 - (train_size+1);
#row = randrange(201, 201+size)
#X_predict = np.array([dataset[row, 0:8]]) #np.array([[0,0,0,0,0,0,0,0]])
X_predict = np.array(dataset[(train_size+1):700, 0:8])

threshold = 0.5

def round_to_threshold(array, threshold):
    i = 0
    while i < size:
        if array[i][0] < threshold:
            array[i][0] = 0
        else:
            array[i][0] = 1
        i = i+1
    return array

#predict
#prediction = [0,0,0,0,0]
#for i in range(len(prediction)):
#    prediction[i] = model.predict(X_predict[i])
prediction = model.predict_on_batch(X_predict)

prediction = round_to_threshold(prediction, threshold)

#print('Predicting for:')
#print(X_predict)

#print('True y value: %f' % dataset[row, 8])
#print(np.array([dataset[(train_size+1):700, 8]]))

#print('Prediction: %f' % prediction)
#print(prediction)
#print(len(prediction))


diff = np.transpose(np.array([dataset[(train_size+1):700, 8]])) - prediction

print(diff)

incorrect = sum(abs(diff))
correct = [700-train_size+1] - incorrect

print('Incorrect: %f' % incorrect[0])
print('Correct: %f' % correct[0])

#print('Percent correct: %f' % (100 * (1 - (incorrect/(700-train_size+1)))))
percent_correct = (100 * (correct / (700-train_size+1)))
print('Percent correct: %f' % percent_correct)


#write prediction results to file
f = open("diabetes_results.txt", "a")
f.write("train size: %f, epochs: %f, batch size: %f, percent_correct: %f \n" % (train_size, epochs, batch_size, percent_correct))
f.close()


#prediction results:
# batch_size        epochs      training examples       prediction examples         % correct
# 20                1           400                     300                         30
# 20                10          400                     300                         61,62
# 20                20          400                     300                         58, 51, 71, 67
# 20                30          400                     300                         69, 68, 69
# 20                40          400                     300                         68
# 20                100         400                     300                         75, 77
# 20                150         400                     300                         72
# 40                150         500                     200                         73, 72, 72
# 20                150         500                     200                         71





#plot metrics
#pyplot.plot(history.history['mean_squared_error'])
#pyplot.plot(history.history['accuracy'])
#pyplot.plot(history.history['difference'])
#pyplot.plot(history.history['true'])
#pyplot.plot(history.history['pred'])
pyplot.plot(history.history['loss'])
pyplot.legend(['loss'])
pyplot.show()













