# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 16:01:09 2020

@author: shubh
"""

# importing modules
import pandas as pd
import pytesseract as pt
import pdf2image
import pyarrow.parquet as pq
import numpy as np
import pyarrow as pa
import sys
import cv2
import pytesseract
import PIL
'''
tesseract_cmd = 'tesseract' 
tesseract_cmd = 'C:\Program Files\Tesseract-OCR/tesseract.exe'

pages = pdf2image.convert_from_path(pdf_path='154_01_1_3_1_Well_Resume.pdf',poppler_path=r'C:\Program Files\poppler-0.68.0\bin', dpi=200, size=(1654,2340))

for i in range(len(pages)):
    pages[i].save('images\\54_01_1_3_1_Well_Resume' + str(i) + '.jpg')
'''
# reading image using opencv
image = cv2.imread('D:/Centrale Supelec/BDRP/BDRP_Project/BDRP_Project-master/images/54_01_1_3_1_Well_Resume8.jpg')
#converting image into gray scale image
#print(str(image))
#image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
image = cv2.fastNlMeansDenoising(image,None,10,7,21)
#image = cv2.bilateralFilter(image,9,75,75)
#image = cv2.medianBlur(image, 3)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# converting it to binary image by Thresholding
# this step is require if you have colored image because if you skip this part 
# then tesseract won't able to detect text correctly and this will give incorrect result
threshold_img = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

#threshold_img = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
# display image
cv2.imshow('threshold image', threshold_img)

result = cv2.imwrite(r'thres_img.png', threshold_img)
if result==True:
  print('File saved successfully')
else:
  print('Error in saving file')

# Maintain output window until user presses a key
cv2.waitKey(0)
# Destroying present windows on screen
cv2.destroyAllWindows()


text_file = open('result'+".txt", "w")

content = pt.image_to_string(threshold_img, lang='eng')

text_file.write(content)
