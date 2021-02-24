from textblob import TextBlob
import os
import re

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


#counting the number of pages / files in extracted_text directory
n = sum(len(files) for _, _, files in os.walk('../BDRP_Project/extracted_text')) #path to the extracted_text file


for i in range(n):
    with open("extracted_text\\img"+str(i)+".txt", "r") as f:        # Opening the test file with the intention to read
        text1 = f.read()                     # Reading the file
        textBlb = TextBlob(text1)            # Making our first textblob
        textCorrected = textBlb.correct()   # Correcting the text
        #corrected_text = remove_special_characters(str(textCorrected))
        os.makedirs("../BDRP_Project/after_spelling_correction", exist_ok=True)
        text_file = open("after_spelling_correction\\text_corrected"+str(i)+".txt", "w")
        text2=str(textCorrected)
        text_file.write(text2)

        originalCompCorrected1 = compare(text1,text2)
        #print("",originalCompCorrected1)
        print("Percentage of fixed mistakes in extracted text\t "+str(i)+"\toafter spelling correction:", percentageOfFixedMistakes(originalCompCorrected1), "%")