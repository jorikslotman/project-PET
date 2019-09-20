import numpy as np
import pydicom
import os
import glob
from PIL import Image
from pydicom.filereader import read_file
import matplotlib.pyplot as plt

input_path = r'C:\Users\CYO-Isala\Documents\DLPET\data\Original'
output_path = r'C:\Users\CYO-Isala\Documents\DLPET\data\Processed\all'



def process_data(input_path,output_path):
    pt_folders = glob.glob(input_path+'\\*')
    for i,folder in enumerate(pt_folders):
        file_list = glob.glob(folder+'\\I*')
        for file in file_list:
            dicom_file = read_file(file)
            pixel_data = dicom_file.pixel_array.astype(np.float)
            pixel_data /= np.max(pixel_data)
            pixel_data *= 255
            p = Image.fromarray(pixel_data.astype(np.uint8))
            p.save(output_path+'\\'+str(i).zfill(3)+os.path.basename(file)+'.bmp')




