import pandas as pd
import pytesseract as pt
import pdf2image
import pyarrow.parquet as pq
import numpy as np
import pyarrow as pa
import sys

pages = pdf2image.convert_from_path(pdf_path='154_01_1_3_1_Well_Resume.pdf',poppler_path=r'C:\Program Files\poppler-0.68.0\bin', dpi=200, size=(1654,2340))

for i in range(len(pages)):
    pages[i].save('images\\54_01_1_3_1_Well_Resume' + str(i) + '.jpg')

for i in range(len(pages)):

    text_file = open(str(i)+".txt", "w")

    content = pt.image_to_string(pages[i], lang='eng')

    text_file.write(content)

