import cv2
import numpy as np
from PIL import Image, ImageOps

#https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv citation for this function
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


def cv_imgs(imgpath):
    img = cv2.imread(imgpath)
    morph = img.copy()
    morph = cv2.fastNlMeansDenoising(img)
    # cv2.imwrite("denoise.png", morph)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 15))
    gray = cv2.cvtColor(morph, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("gray.png", gray)
    gradient_image = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    # cv2.imwrite("morphology.png", gradient_image)

    blur = cv2.medianBlur(gradient_image,3)
    # cv2.imwrite("blur.png", blur)

    ret, thing = cv2.threshold(blur, 5, 255.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


    th3 = cv2.adaptiveThreshold(thing,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY_INV,11,2)
    # cv2.imwrite("threshold.png", th3)


    img_dilation = cv2.dilate(th3, kernel, iterations=4)
    # cv2.imwrite("dilate.png", img_dilation)
    conturs_lst = cv2.findContours(img_dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]


    coor_lst = []
    for cnt in conturs_lst:
        x,y,w,h = cv2.boundingRect(cnt)
        if w < 35 or h < 35:
            continue
        coor_lst.append((x,y,w,h))


    coor_lst.sort(key=lambda tup: tup[0])
    pic_lst = []
    exp_lst = []
    comp = 0
    for coor in coor_lst:
        comp +=1
        x = coor[0]
        y = coor[1]
        w = coor[2]
        h = coor[3]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 255), 1, cv2.LINE_AA)
        roi = 0
        if h > w:
            roi = img[y+30:y+h-30, x+2:x+w-2]
        if h < w:
            roi = img[y+2:y+h-2, x+2:x+w-2]

        exp_stuff = [x,y,w,h]
        exp_lst.append(exp_stuff)
        # cv2.imwrite(str(comp) + '.jpg', roi)
        # cv2.imwrite(str(comp) + 'seeing.jpg', roi)

        # new_img = cv2.imread(str(comp) + '.jpg', cv2.IMREAD_GRAYSCALE)
        #No need to write locally?
        new_img = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)

        thresh, im_bw = cv2.threshold(new_img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        final = cv2.bitwise_not(im_bw)
        resize_img = 0
        if h > w:
            resize_img = resize2SquareKeepingAspectRation(final, h-60, interpolation = cv2.INTER_AREA)
        if h < w:
            resize_img = resize2SquareKeepingAspectRation(final, w-60, interpolation = cv2.INTER_AREA)
        else:
            resize_img = resize2SquareKeepingAspectRation(final, h, interpolation = cv2.INTER_AREA)
        dilation = cv2.dilate(resize_img,np.ones((5,5),np.uint8),iterations = 1)
        resize_img = resize2SquareKeepingAspectRation(dilation, 26, interpolation = cv2.INTER_AREA)
        constant = cv2.copyMakeBorder(resize_img, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=[0,0,0])

        # cv2.imwrite(str(comp) + '.jpg', constant)
        pic_lst.append(constant)
    cv2.imwrite("bounded.png", img)
    return pic_lst, exp_lst
