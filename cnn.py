import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from keras.models import model_from_yaml
from sklearn.utils import class_weight

curr_url = "https://www.kaggle.com/hupe1980/keras-digit-recognizer-mnist-data"
def load_data():
    df_train  = pd.read_csv("shuffleonevsslash.csv")

    y_train = df_train['label'].values
    X_train = df_train.drop(columns=['label']).values

    return X_train, y_train



X_train, y_train = load_data()

class_weight = class_weight.compute_class_weight('balanced',np.unique(y_train),y_train)

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32')

input_shape = X_train.shape[1:]

X_train = X_train

y_train = to_categorical(y_train)

num_classes = y_train.shape[1]







def convolutional_model(num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy',  metrics=['accuracy'])
    return model

model = convolutional_model(num_classes)
model.fit(X_train, y_train, validation_split=0.3, epochs=4, batch_size=128, verbose=1, class_weight=class_weight)


model_yaml = model.to_yaml()
with open("slashvsone.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
# serialize weights to HDF5
model.save_weights("slashvsone.h5")
print("Saved model to disk")

# im = cv2.imread('11.jpg',0)
# im = cv2.resize(im,  (28, 28))
# im.reshape((28,28)) # (28,28)
#
# batch = np.expand_dims(im,axis=0)# (1, 28, 28)
# batch = np.expand_dims(batch,axis=3) # (1, 28, 28,1)
#
#
# print(model.predict(batch, batch_size=1))
