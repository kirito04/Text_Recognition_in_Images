import numpy as np
import cv2


def preprocessing(img):

    # resize image to (300, 300)
    img = set_image_dpi(img)
    # image to binary
    img = deskew(img)
    return img


def set_image_dpi(img):
    length_x, width_y, _ = img.shape
    im_resized = cv2.resize(img, (1000, 360), interpolation=cv2.INTER_AREA)
    return im_resized


def to_binary(img):
    # convert the image to grayscale and flip the foreground
    # and background to ensure foreground is now "white" and
    # the background is "black"
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    # threshold the image, setting all foreground pixels to
    # 255 and all background pixels to 0
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return img

def deskew(img):
    img = to_binary(img)
    # grab the (x, y) coordinates of all pixel values that
    # are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all
    # coordinates
    coords = np.column_stack(np.where(img > 0))
    angle = cv2.minAreaRect(coords)[-1]
    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    print(angle)
    return rotated


img = cv2.imread("photo3.png")
print(img.shape)
img = preprocessing(img)

cv2.imshow("Preprocessed Image", img)
cv2.waitKey(0)

import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
print(pytesseract.image_to_string(img))

boxes = pytesseract.image_to_data(img)
for a,b in enumerate(boxes.splitlines()):
        print(b)
        if a!=0:
            b = b.split()
            if len(b)==12:
                x,y,w,h = int(b[6]),int(b[7]),int(b[8]),int(b[9])
                cv2.putText(img,b[11],(x,y-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(50,50,255),1)
                cv2.rectangle(img, (x,y), (x+w, y+h), (50, 50, 255), 2)

cv2.imshow("image", img)
cv2.waitKey(0)