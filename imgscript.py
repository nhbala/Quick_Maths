from scipy import ndimage, misc
import numpy as np
import os
import cv2
import pandas as pd


path = "/Users/nathanbala/Downloads/extracted_images/-"

index_value_dict = {
    "10": ")",
    "11": "(",
    "12": "+",
    "13": "-",
    "14": "*",
    "15": "a",
    "16": "c",
    "17": "n",
    "18": "="
}
#do 19 and 20
dict = {
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


def resize2SquareKeepingAspectRation(img, size, interpolation):
  h, w = img.shape[:2]
  c = None if len(img.shape) < 3 else img.shape[2]
  if h == w: return cv2.resize(img, (size, size), interpolation)
  if h > w: dif = h
  else:     dif = w
  x_pos = int((dif - w)/2.)
  y_pos = int((dif - h)/2.)
  if c is None:
    mask = np.zeros((dif, dif), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
  else:
    mask = np.zeros((dif, dif, c), dtype=img.dtype)
    mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
  return cv2.resize(mask, (size, size), interpolation)


large_array = []
for image_path in os.listdir(path):
    input_path = os.path.join(path, image_path)
    image_to_process = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((5,5),np.uint8)
    thresh, im_bw = cv2.threshold(image_to_process, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    final = cv2.bitwise_not(im_bw)

    dilation = cv2.dilate(final,np.ones((5,5),np.uint8),iterations = 1)
    # constant = cv2.copyMakeBorder(dilation, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0,0,0])
    resize_img = resize2SquareKeepingAspectRation(dilation, 26, interpolation = cv2.INTER_AREA)
    constant = cv2.copyMakeBorder(resize_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0,0,0])
    cv2.imwrite("steve.jpg", resize_img)
    cv2.imwrite("steve1.jpg", constant)
    final = np.concatenate(constant, axis=0 )
    final = np.concatenate([[1],final])
    final = final.astype(int)
    large_array.append(final)

f=open('equalsminus.csv','ab')
np.savetxt(f,large_array, delimiter=',', fmt='%u')

# df_train  = pd.read_csv("finalfuck.csv")
# my_list = df_train["label"].values
# uniqueVals = np.unique(my_list)
# print(uniqueVals)


    # input_path = os.path.join(path, image_path)
    # image_to_rotate = ndimage.imread(input_path)
