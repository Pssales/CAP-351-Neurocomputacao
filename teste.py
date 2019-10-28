import numpy as np 
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
# from loss_plot import loss_plot
from keras.optimizers import Adam

def loss_plot(history):
    train_acc = history.history['val_loss']
    val_acc = history.history['val_accuracy']

    plt.plot(np.arange(1,21), train_acc, marker = 'D', label = 'Loss Accuracy')
    plt.xlabel('epocas')
    plt.ylabel('acuracia')
    plt.plot('Train/Validation')
    plt.legend()
    plt.margins(0.02)
    plt.show()


epochs = 20
batch_size = 128
optimizer = Adam(lr=0.0001)
input_shape = (28,28,1)

(X_train,y_train),(X_test,y_test) = mnist.load_data()
X_train,X_val,y_train, y_val = train_test_split(X_train, y_train,stratify = y_train,test_size = 0.08333)

X_train = X_train.reshape(-1,784)
X_val = X_val.reshape(-1,784)
X_test = X_test.reshape(-1,784)

model = Sequential()
model.add(Dense(300,input_shape=(784,), activation = 'relu'))
model.add(Dense(300, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.compile(loss = 'sparse_categorical_crossentropy',optimizer = optimizer, metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size, validation_data = (X_val, y_val))

loss, acc = model.evaluate(X_test, y_test)
print('Test loss: ', loss)
print('Accuracy: ', acc)
print(history.history)
loss_plot(history)