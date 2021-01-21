from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import numpy as np
import cv2
import pdf2image

pages = pdf2image.convert_from_path(pdf_path='doc.pdf',poppler_path=r'C:\Program Files\poppler-0.68.0\bin', dpi=200, size=(1654,2340))
n=len(pages)

def compare_images(imageA,imageB):
    s = ssim(imageA,imageB,multichannel=True)
    return s

for i in range(n):
    original_image = cv2.imread('img\\image' + str(i) + '.jpg')
    original_image = cv2.resize(original_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    preprocessed_image = cv2.imread('preprocessed_images\\preprocessed_img' + str(i) + '.jpg')
    print("SSIM for image "+str(i)+" is:",compare_images(original_image,preprocessed_image))
    #print(original_image.shape)
    #print(preprocessed_image.shape)



