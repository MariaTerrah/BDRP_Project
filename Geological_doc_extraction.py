# -*- coding: utf-8 -*-



import pytesseract as pt
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as im
from scipy.ndimage import interpolation as inter
import cv2
import pdf2image
import pathlib
from textblob import TextBlob

import re

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from pathlib import Path

import findspark
findspark.find()
findspark.init()

import pyspark
import random
sc = pyspark.SparkContext(appName="topic_mod")
print("Initialization successful")

import operator;
import re;


#final

def find_score(arr, angle):
    data = inter.rotate(arr, angle, reshape=False, order=0)
    hist = np.sum(data, axis=1)
    score = np.sum((hist[1:] - hist[:-1]) ** 2)
    return hist, score
# function to remove special characters
def remove_special_characters(text):
    # define the pattern to keep
    pat = r'[^a-zA-z0-9.,!?/:;\"\'\s]'
    return re.sub(pat, '', text)


#Metrics to evaluate the spelling correction
def compare(text1, text2):
    l1 = text1.split()
    l2 = text2.split()
    good = 0
    bad = 0
    for i in range(0, len(l1)):
        if l1[i] != l2[i]:
            bad += 1
        else:
            good += 1
    return (good, bad)

# Helper function to calculate the percentage of correctly spelled words
def percentageOfFixedMistakes(x):
    if x[0]+x[1] == 0 :
        return 0
    else:
        return (x[1] / (x[0] + x[1])) * 100

#transforming pdf file to images

#insert pdf file path in the pdf_path
pages = pdf2image.convert_from_path(pdf_path='data\\10_8-1_10-08-01-pb-707-0308.pdf',poppler_path=r'C:\Program Files\poppler-0.68.0\bin', dpi=200, size=(1654,2340))
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


path = Path(__file__).parent / "./skew_corrected_images"
count = 0
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
    
    count +=1
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
        
        rect = 0
        for x in boxes:
            rect +=1
            #print (rect)
        if rect > 20:
            print ("Exist")
            #cv2.imwrite(path+'/tab_det_plane'+sample+'.jpg', inverse)
        else:
            print ("Does not exist")
            cv2.imwrite(r'preprocessed_images\\preprocessed_img' + str(i) + '.jpg', read_image)
            os.makedirs("../BDRP_Project/extracted_text", exist_ok=True)
            text_file = open("extracted_text\\img" + str(count) + ".txt", "w")
            content = pt.image_to_string(read_image, lang='eng')

            text_file.write(content)
        return (num, boxes)
    cont, boxes = get_boxes(cont, method="top-to-bottom")

    count_coord = 0
#counting the number of pages / files in extracted_text directory


for i in range(n):
    if os.path.exists("extracted_text\\img"+str(i)+".txt"):
        with open("extracted_text\\img" + str(i) + ".txt","r") as f:  # Opening the test file with the intention to read
            text1 = f.read()  # Reading the file
            textBlb = TextBlob(text1)  # Making our first textblob
            textCorrected = textBlb.correct()  # Correcting the text
            # corrected_text = remove_special_characters(str(textCorrected))
            os.makedirs("../BDRP_Project/after_spelling_correction", exist_ok=True)
            text_file = open("after_spelling_correction\\text_corrected" + str(i) + ".txt", "w")
            text2 = str(textCorrected)
            text_file.write(text2)
            #evaluating
            originalCompCorrected1 = compare(text1, text2)
            # print("",originalCompCorrected1)
            print("Percentage of fixed mistakes in extracted text\t " + str(i) + "\t after spelling correction:",
                  percentageOfFixedMistakes(originalCompCorrected1), "%")
    else:
        pass



    
'''
Most Frequent Words
'''  
path = Path(__file__).parent / "./extracted_text"

for sample in os.listdir(path):
    input_path = os.path.join(path, sample)

    def preprocess(text):
        '''
        Regular expression for removing all non-letter characters in the file.
        '''
        regex = re.compile('[^a-zA-Z ]')
        '''
        Step 1. 
        Remove the non-letter characters.
        '''
        text = text.map(lambda line: regex.sub('', line))
        
        '''
        Step 2.
        Apply a transformation on the RDD text to obtain an RDD words, where each element is a word from the file.
        '''
        words = text.flatMap(lambda line: line.split(' '))
        
        '''
        Step 3.
        Filter out from the RDD words the words with length 0.
        
        '''
        words = words.filter(lambda word: len(word) > 0)
        
        '''
        Step 4.
        Lowercase all words of the RDD words
        
        '''
        words = words.map(lambda word: word.lower())
        
        '''
        Step 5.
        Returns the RDD words.
        '''
        return words
    
    
    # Reads the novel into a RDD
    geo_doc = sc.textFile(str(input_path))
    
    words = preprocess(geo_doc)
    
    print("Number of words ", words.count())
    
    '''
    Top-100 most frequent words
    '''
    # countByValue() counts the number of occurrences of each word.
    # The list will be sorted in ascending order of number of occurrences
    occurrences = sorted(words.countByValue().items(), key=operator.itemgetter(1))
    
    # Reverse the order
    occurrences.reverse()
    print("Top-50 most frequent words")
    i = 0
    for (w, f) in occurrences:
        i += 1
        if i > 50:
            break
        print(w,"--->",f)
        
    def preprocess(text, stopwords):
        '''
        Regular expression for removing all non-letter characters in the file.
        '''
        regex = re.compile('[^a-zA-Z ]')
        '''
        Step 1. 
        Remove the non-letter characters.
        '''
        text = text.map(lambda line: regex.sub('', line))
        
        '''
        Step 2.
        Obtain the RDD containing the words in the file.
        '''
        words = text.flatMap(lambda line: line.split(" "))
    
        '''
        Step 3.
        Filter out the words with length 0.
        '''
        words = words.filter(lambda word: len(word) > 0)
        
        '''
        Step 4.
        Lowercase all words of the RDD $words$
        '''
        words = words.map(lambda word: word.lower())
        
        '''
        Step 5 Remove stop words.
        '''
        words = words.subtract(stopwords)
        
        
        # Returns the words
        return words
    
    
    # Load the stopwords to an RDD
    stop_path= os.getcwd()
    stopwords = sc.textFile(str(stop_path)+"./stopwords.txt")
    words = preprocess(geo_doc, stopwords)
    print("Number of words after stopword removal ", words.count())
    
    # Count the number of occurrences as before.
    occurrences = sorted(words.countByValue().items(), key=operator.itemgetter(1))
    # Reverse the order
    occurrences.reverse()
    print("Top-50 most frequent words")
    i = 0
    for (w, f) in occurrences:
        i += 1
        if i > 50:
            break
        #print(w, ", ", f)
        
    kvwords = words.map(lambda word : (word, 1))
    print(kvwords.take(100))
    
    occurrences = kvwords.reduceByKey(lambda x, y : x + y)
    occurrences = occurrences.sortBy(lambda x: x[1], ascending=False)
    occurrences.take(100)
    
    occurrences = kvwords.groupByKey()
    occurrences = occurrences.map(lambda x : (x[0], len(x[1])))
    occurrences = occurrences.sortBy(lambda x: x[1], ascending=False)
    occurrences.take(100)
    
    occurrences = sorted(kvwords.countByKey().items(), key=operator.itemgetter(1))
    # Reverse the order
    occurrences.reverse()
    
    word_freq = {}
    print("Top-15 most frequent words")
    i = 0
    for (w, f) in occurrences:
        i += 1
        if i > 15:
            break
        print(w, ", ", f)
        word_freq[w]= f
    print(word_freq)
    
    '''
    Plotting Frequent Words
    ''' 
    
    cloud = WordCloud(max_font_size=80,colormap="hsv").generate_from_frequencies(word_freq)
    plt.figure(figsize=(16,12))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    #plt.show()
    
    os.makedirs("../BDRP_Project/frequent_word_images", exist_ok=True)
    plt.savefig('frequent_word_images\\cloud_img' + str(i)+'.png')
    
    #word_freq.plot(30,title='Frequency distribution for 30 most common tokens in our text collection (excluding stopwords and punctuation)')
    
    plt.bar(range(len(word_freq)), list(word_freq.values()), align='center')
    plt.xticks(range(len(word_freq)), list(word_freq.keys()))
    plt.rcParams["figure.figsize"] = (20,10)
    plt.title('Top 15 most frequent words')
    #plt.show()
    os.makedirs("../BDRP_Project/frequent_word_images", exist_ok=True)
    plt.savefig('frequent_word_images\\graph_img' + str(i)+'.png')
