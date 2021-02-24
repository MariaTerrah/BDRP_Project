import os
import pytesseract as pt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import cv2
import pdf2image
import pathlib



#final

def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score

#transforming pdf file to images

#insert pdf file path in the pdf_path
pages = pdf2image.convert_from_path(pdf_path='data\\doc.pdf',poppler_path=r'C:\Program Files\poppler-0.68.0\bin', dpi=200, size=(1654,2340))
n=len(pages)

for i in range(len(pages)):

    #First skew correction
    os.makedirs("../BDRP_Project/img", exist_ok=True) #create directory to save the images
    img = 'img\\image' + str(i) + '.jpg'
    pages[i].save(img)

    img = im.open(img)
    # convert to binary

    wd, ht = img.size
    pix = np.array(img.convert('1').getdata(), np.uint8)
    bin_img = 1 - (pix.reshape((ht, wd)) / 255.0)
    #plt.imshow(bin_img, cmap='gray')
    #plt.savefig('binary.png')


    delta = 1
    limit = 5
    angles = np.arange(-limit, limit+delta, delta)
    scores = []

    for angle in angles:
        hist, score = find_score(bin_img, angle)
        scores.append(score)
    best_score = max(scores)
    best_angle = angles[scores.index(best_score)]
    print('Best angle for image' +str(i)+': {}'.format(best_angle))
    # correct skew
    data = inter.rotate(bin_img, best_angle, reshape=False, order=0)
    img_s = im.fromarray(255-(255 * data).astype("uint8")).convert("RGB")
    os.makedirs("../BDRP_Project/skew_corrected_images", exist_ok=True)
    img_s.save(r'skew_corrected_images\\skew_corrected_img' + str(i) + '.jpg')


    # img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_c='skew_corrected_images\\skew_corrected_img' + str(i) + '.jpg'
    img1 = cv2.imread(img_c,cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # rescaling image
    img1 = cv2.resize(img1, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    #img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    # noise removal
    img2 = cv2.GaussianBlur(img1, (5, 5), 0)

    # thresholding using otsu method
    ret3, img2 = cv2.threshold(img2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #save preprpcessed image
    os.makedirs("../BDRP_Project/preprocessed_images", exist_ok=True)
    img_prep = cv2.imwrite(r'preprocessed_images\\preprocessed_img' + str(i) + '.jpg', img2)

    os.makedirs("../BDRP_Project/extracted_text", exist_ok=True)
    text_file = open("extracted_text\\img" + str(i) + ".txt", "w")

    content = pt.image_to_string(img2, lang='eng')

    text_file.write(content)






