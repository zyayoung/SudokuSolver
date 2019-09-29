from __future__ import print_function
import cv2
import numpy as np

x_ext = []
y_ext = []
for size in range(16,30):
    for font in [cv2.FONT_ITALIC, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL, cv2.FONT_HERSHEY_COMPLEX]:
        for i in range(1,10):
            dig = np.zeros((32,32))
            cv2.putText(dig, str(i), (2,28), font, 2/28*size, color=(255,255,255))
            cols = dig.mean(axis=0)>1
            dig = dig[dig.mean(axis=1)>1,:]
            dig = dig[:,cols]
            x_ext.append(cv2.resize(dig, (28, 28), interpolation=cv2.INTER_LINEAR) > 127)
            y_ext.append(i)
x_ext = np.array(x_ext)
y_ext = np.array(y_ext, dtype=int)

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

for i in range(len(x_train)):
    dig = x_train[i].copy()
    cols = dig.mean(axis=0)>1
    dig = dig[dig.mean(axis=1)>1,:]
    dig = dig[:,cols]
    x_train[i] = cv2.resize(dig, (28, 28), interpolation=cv2.INTER_LINEAR) > 127
for i in range(len(x_test)):
    dig = x_test[i].copy()
    cols = dig.mean(axis=0)>1
    dig = dig[dig.mean(axis=1)>1,:]
    dig = dig[:,cols]
    x_test[i] = cv2.resize(dig, (28, 28), interpolation=cv2.INTER_LINEAR) > 127

x_train = np.concatenate([x_train, x_ext])
y_train = np.concatenate([y_train, y_ext])

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    x_ext = x_ext.reshape(x_ext.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_ext = x_ext.reshape(x_ext.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_ext = x_ext.astype('float32')
x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_ext = keras.utils.to_categorical(y_ext, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu', strides=2,
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), strides=2, activation='relu'))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
model.fit(x_ext, y_ext,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
model.save("model.h5")
print('Test loss:', score[0])
print('Test accuracy:', score[1])
