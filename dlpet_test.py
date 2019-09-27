import numpy as np
import pydicom
import os
import glob
from PIL import Image
from pydicom.filereader import read_file
import matplotlib.pyplot as plt


def preprocess_single_image(x):
    x /= np.max(x)
    x *= 255
    x = x.astype(np.uint8)
    return x



def process_data(input_path,output_path):
    # find all folders in input_path
    pt_folders = glob.glob(input_path+'\\*')
    # iterate over found folders
    for scan_nr,folder in enumerate(pt_folders):

        print('Started folder ' + str(scan_nr) + ' / ' + str(len(pt_folders)) )

        # read patient_number:
        p = folder.find('_pat')
        p2 = p+1+folder[p+1:].find('_')
        pat_nr = int(folder[p+4:p2])


        # find dose of folder
        if folder.find('halfdose') > 0:
            dose = 'half'
        else:
            dose = 'full'

        # find all DICOM slices in current folder
        file_list = glob.glob(folder+'\\I*')

        # iterate over folders
        for slice_nr,file in enumerate(file_list):
            I_nr = os.path.basename(file)[1:]
            I_nr = 'I'+I_nr.zfill(4)

            # read current file and retrieve pixel_data
            dicom_file = read_file(file)
            pixel_data = dicom_file.pixel_array.astype(np.float)

            # preprocess slice
            x = preprocess_single_image(pixel_data)

            # make image
            p = Image.fromarray(x)
            file_name = 'pat_'+str(pat_nr).zfill(4)+'_'+I_nr+'_'+dose+'.bmp'

            p.save(output_path+'\\'+dose+'\\'+file_name)





