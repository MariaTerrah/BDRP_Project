import cv2
import numpy as np
from PIL import ImageOps
from PIL import Image
import os


n = sum(len(files) for _, _, files in os.walk('../BDRP_Project/extracted_text')) #path to the extracted_text file


for i in range(n):
    image = cv2.imread('img\\image'+str(i)+'.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    dilate = cv2.dilate(thresh, kernel, iterations=4)

    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36,255,12), 2)

# cv2.imshow('th', thresh)
# cv2.imshow('dilated', dilate)
# cv2.imshow('image', image)
# cv2.waitKey()

    img = Image.fromarray(image, 'RGB')
    os.makedirs("../BDRP_Project/text_box_detection", exist_ok=True)
    img.save(r'text_box_detection\\img_with_boxes'+str(i)+'.jpg')