# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 23:33:27 2021

@author: shubh
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path='D:/Centrale Supelec/BDRP/BDRP_Project/BDRP_Project-master/Image'
#sample=r'D:/Centrale Supelec/BDRP/BDRP_Project/BDRP_Project-master/images/54_01_1_3_1_Well_Resume8.jpg'
for sample in os.listdir(path):
    input_path = os.path.join(path, sample)
    read_image= cv2.imread(str(input_path),0)

    convert_bin,grey_scale = cv2.threshold(read_image,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    grey_scale = 255-grey_scale
    grey_graph = plt.imshow(grey_scale,cmap='Blues')
    #plt.show()

    length = np.array(read_image).shape[1]//100
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))

    horizontal_detect = cv2.erode(grey_scale, horizontal_kernel, iterations=3)
    hor_line = cv2.dilate(horizontal_detect, horizontal_kernel, iterations=3)
    plotting = plt.imshow(horizontal_detect,cmap='Blues')
    plt.show()
    #cv2.imwrite('tab_det_3_hori.jpg', hor_line)

    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    vertical_detect = cv2.erode(grey_scale, vertical_kernel, iterations=3)
    ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=3)
    show = plt.imshow(vertical_detect,cmap='Blues')
    plt.show()
    #cv2.imwrite('tab_det_3_ver.jpg', ver_lines)

    final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combine = cv2.addWeighted(ver_lines, 1, hor_line, 1, 0.0)
    combine = cv2.erode(~combine, final, iterations=2)
    thresh, combine = cv2.threshold(combine,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #convert_xor = cv2.bitwise_xor(read_image,combine)
    convert_xor = cv2.bitwise_not(combine)
    inverse = cv2.bitwise_not(convert_xor)
    output= plt.imshow(convert_xor,cmap='Blues')
    plt.show()
    cv2.imwrite(path+'/tab_det_plane'+sample+'.jpg', inverse)

    cont, _ = cv2.findContours(combine, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    def get_boxes(num, method="left-to-right"):
        invert = False
        flag = 0
        if method == "right-to-left" or method == "bottom-to-top":
            invert = True
            #print("Exist1")
        if method == "top-to-bottom" or method == "bottom-to-top":
            flag += 1
            #print("Exist2")
        boxes = [cv2.boundingRect(c) for c in num]
        (num, boxes) = zip(*sorted(zip(num, boxes),
        key=lambda b:b[1][1], reverse=invert))
        print (boxes)
        print (flag)
        rect = 0
        for x in boxes:
            rect +=1
            print (rect)
        if rect > 1:
            print ("Exist")
        else:
            print ("Does not")
        return (num, boxes)
    cont, boxes = get_boxes(cont, method="top-to-bottom")