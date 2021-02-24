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
from PIL import Image
from pathlib import Path


path = Path(__file__).parent / "./img"
count_total = []
#sample=r'D:/Centrale Supelec/BDRP/BDRP_Project/BDRP_Project-master/images/54_01_1_3_1_Well_Resume8.jpg'
for sample in os.listdir(path):
    input_path = os.path.join(path, sample)
    read_image= cv2.imread(str(input_path),0)
    
    #convert image to binary
    convert_bin,grey_scale = cv2.threshold(read_image,128,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    grey_scale = 255-grey_scale
    grey_graph = plt.imshow(grey_scale,cmap='Blues')
    #plt.show()

    length = np.array(read_image).shape[1]//100
    
    #create kernel for horizontal lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (length, 1))
    
    horizontal_detect = cv2.erode(grey_scale, horizontal_kernel, iterations=3)
    hor_line = cv2.dilate(horizontal_detect, horizontal_kernel, iterations=3)
    plotting = plt.imshow(horizontal_detect,cmap='Blues')
    plt.show()
    #cv2.imwrite('tab_det_3_hori.jpg', hor_line)
    
    #create kernel for vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, length))
    vertical_detect = cv2.erode(grey_scale, vertical_kernel, iterations=3)
    ver_lines = cv2.dilate(vertical_detect, vertical_kernel, iterations=3)
    show = plt.imshow(vertical_detect,cmap='Blues')
    plt.show()
    #cv2.imwrite('tab_det_3_ver.jpg', ver_lines)
    
    #combining both the horizontal and vertical lines
    final = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    combine = cv2.addWeighted(ver_lines, 1, hor_line, 1, 0.0)
    combine = cv2.erode(~combine, final, iterations=2)
    thresh, combine = cv2.threshold(combine,128,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #convert_xor = cv2.bitwise_xor(read_image,combine)
    convert_xor = cv2.bitwise_not(combine)
    inverse = cv2.bitwise_not(convert_xor)
    output= plt.imshow(convert_xor,cmap='Blues')
    plt.show()
    #cv2.imwrite(path+'/tab_det_plane'+sample+'.jpg', inverse)
    
    
    #detect cells from the image
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
        #print (boxes)
        #print (flag)
        box = [list(row) for row in boxes]
        print(box)
        
        count_coord = 0
        count_total = 0
        
        count_coord = len(box)
        count_total = count_total+count_coord
        
        #for list1 in box:
            #count_coord = count_coord + int(list1)
        #print (count_coord)
        #print (count_total)
            #print (list1[2])
            #new_image = cv2.rectangle(read_image, (list1[0],list1[1]),(list1[0]+list1[2],list1[1]+list1[3]), (0,0,0),-1)
            #new_op= plt.imshow(new_image)
            #new_image = cv2.rectangle(read_image, start_point, end_point, color, -1)
            #new_op= plt.imshow(new_image)
            #plt.show()
            #img_arr = np.array(read_image)
            #img_arr[list1[0] : list1[1], list1[2] : list1[3]] = (0, 0, 0)
            #img = Image.fromarray(img_arr)
            #img.show()
        
        rect = 0
        for x in boxes:
            rect +=1
            #print (rect)
        if rect > 20:
            print ("Exist")
            #cv2.imwrite(path+'/tab_det_plane'+sample+'.jpg', inverse)
        else:
            print ("Does not exist")
            cv2.imwrite(str(path)+'/image'+sample+'.jpg', read_image)
        return (num, boxes)
    cont, boxes = get_boxes(cont, method="top-to-bottom")

    count_coord = 0
 
    '''
    
    box = [list(row) for row in boxes]
    count_coord = len(box)
    print (count_coord)
    count_total.append(count_coord)


print (count_total)
avg_num_cells = sum(count_total)/len(count_total)
print (avg_num_cells)

file_len = len(os.listdir(path))
print (file_len)

index=0
for i in count_total:
    if i > avg_num_cells:
        continue
    else:
        for sample in os.listdir(path):
            input_path = os.path.join(path, sample)
            read_image= cv2.imread(str(input_path),0)
            sam_index= sample.index(sample)
            print(sam_index)
            if sam_index== count_total.index(i):
                cv2.imwrite(path+'/tab_det_plane'+sample+'.jpg', read_image)
                
    #for sample in os.listdir(path):
        #input_path = os.path.join(path, sample)
        #read_image= cv2.imread(str(input_path),0)
        #for j in count_total:
            #if index == count_total.index(j):
                #if j > avg_num_cells:
                    #continue
                #else:
                    #cv2.imwrite(path+'/tab_det_plane'+sample+'.jpg', read_image)

            #if i > avg_num_cells:
            #continue
        #else:
            #cv2.imwrite(path+'/tab_det_plane'+sample+'.jpg', read_image)
    
'''
    