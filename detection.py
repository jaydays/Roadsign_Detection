#%%

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, AveragePooling2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam


#%%

input_shape = (32, 32, 3)
numCategories = 47 #47 different signs

cnn = Sequential()


##Convolutional Layers
cnn.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
cnn.add(Conv2D(filters=16, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid',data_format=None))
#cnn.add(Dropout(0.5))

cnn.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid',data_format=None))
#cnn.add(Dropout(0.5))

cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(MaxPool2D(pool_size=(2,2), strides=2, padding='valid',data_format=None))
#cnn.add(Dropout(0.5))

cnn.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu'))
cnn.add(BatchNormalization())
cnn.add(AveragePooling2D(pool_size=(2,2), strides=2, padding='valid',data_format=None))
#cnn.add(Dropout(0.5))


##Fully Connected
cnn.add(Flatten())

#cnn.add(Dense(512, activation='sigmoid'))
#cnn.add(BatchNormalization())
#cnn.add(Dropout(0.5))

cnn.add(Dense(512, activation='sigmoid'))
#cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))


##Output
cnn.add(Dense(numCategories, activation='softmax'))


print(cnn.summary())
adam = Adam(lr=0.001)
cnn.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

#%%

batch_size = 64
epochs = 100

cnn.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs, verbose=2)

