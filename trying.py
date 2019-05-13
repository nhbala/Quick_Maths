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
from one import gradient


index_value_dict = {
    "0": "0",
    "1": "1",
    "2": "2",
    "3": "3",
    "4": "4",
    "5": "5",
    "6": "6",
    "7": "7",
    "8": "8",
    "9": "9",
    "10": ")",
    "11": "(",
    "12": "+",
    "13": "-",
    "14": "*",
    "15": "a",
    "16": "c",
    "17": "n",
    "18": "=",
    "19": "e",
    "20": "pi",
    "21": "/"
}

equals_minus_dict = {
    "0": "=",
    "1": "-"
}


def get_rep(image):
    yaml_file = open('models/finalmodel.yaml', 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    loaded_model.load_weights("models/finalmodel.h5")

    # yaml_file = open('test1.yaml', 'r')
    # loaded_model_yaml = yaml_file.read()
    # yaml_file.close()
    # loaded_model = model_from_yaml(loaded_model_yaml)
    # loaded_model.load_weights("test1.h5")


    im = image
    im = cv2.resize(im,  (28, 28))
    im.reshape((28,28))

    batch = np.expand_dims(im,axis=0)
    batch = np.expand_dims(batch,axis=3)

    final = loaded_model.predict(batch, batch_size=1)
    lmao = max(final[0])
    result = [i for i, j in enumerate(final[0]) if j == lmao]
    return_value = index_value_dict[str(result[0])]
    if return_value == "1":
        return_value = gradient(im)
    if return_value == "=" or return_value == "-":
        yaml_file = open('models/minusequals.yaml', 'r')
        loaded_model_yaml = yaml_file.read()
        yaml_file.close()
        loaded_model = model_from_yaml(loaded_model_yaml)
        loaded_model.load_weights("models/minusequals.h5")
        final = loaded_model.predict(batch, batch_size=1)
        lmao = max(final[0])
        result = [i for i, j in enumerate(final[0]) if j == lmao]
        return_value = equals_minus_dict[str(result[0])]

    return return_value
