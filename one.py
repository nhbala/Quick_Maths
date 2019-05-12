import cv2 as cv

# START -> GRADIENT <- START #
def gradient(file):
    img = cv.fastNlMeansDenoising(file) #denoising
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    sobely= cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
    angles = cv.phase(sobelx,sobely,angleInDegrees=True)
    angle = 0
    totalM = 0
    for y in range(28):
        for x in range(28):
            magnitude = (sobelx[y][x]**2 + sobely[y][x]**2)**(0.5)
            totalM += magnitude
            angle += magnitude * angles[y][x]
    print(angle/totalM)
    if angle/totalM > 145.0:
        return "1"
    else:
        return "/"
# END -> GRADIENT <- END #
