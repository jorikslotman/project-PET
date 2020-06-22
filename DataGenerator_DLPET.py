import os
import glob
from PIL import Image
from cv2 import resize, INTER_CUBIC
import random as rn

import numpy as np

from tensorflow.keras.utils import Sequence, to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataGenerator(Sequence):

    # folder with classes separated in sub directories
    # only for binary so far! (no categorial)

    def __init__(
            self, 
            folder, 
            batch_size = 32, 
            dim = (288,288),
            n_channels = 1,
            shuffle = True
            ):

        self.dim = dim
        self.batch_size = batch_size
        self.folder = folder
        self.n_channels = n_channels
        self.shuffle = shuffle


        file_list = []
        class_list = []

        for d in glob.glob(os.path.join(self.folder,'*','')):
            file_list_t = glob.glob(os.path.join(d,'*.bmp'))
            file_list.extend(file_list_t)
            class_list.extend([os.path.basename(os.path.dirname(d))] * len(file_list_t))

        self.data_list = np.array([file_list,class_list])
        self.classes = np.unique(self.data_list[1,:])

        self.on_epoch_end()

# bepaalt aantal stappen per epoch (totaal aantal files / badge grootte
    def __len__(self):

        return int(np.ceil(self.data_list.shape[1] / self.batch_size))


    def __getitem__(self,index):

        if index > np.floor(self.data_list.shape[1] / self.batch_size): # incomplete batch
            batch_indexes = self.indexes[index * self.batch_size:] # grab the last ones from the list
        else:
            batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size] # pakt volgende comlete batch

        batch_list = self.data_list[:,batch_indexes]
        self.batch_list = batch_list

        X, y = self.__data_generation(batch_list)

        return X, y
        # X = [32,512,512,1]
        # y =[32] bijv: [0,0,1,0,1,1,0,...]



    def on_epoch_end(self):

        self.indexes = np.arange(self.data_list.shape[1])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self,batch_list):

        X = np.empty((batch_list.shape[1], *self.dim,self.n_channels),dtype='float')
        y = np.empty((batch_list.shape[1]), dtype='|U16')

        for i,file_t in enumerate(batch_list[0,:]):

            X_t = np.array(Image.open(file_t,))

            # resize to self.dim
            X_t = resize(X_t, dsize=self.dim, interpolation=INTER_CUBIC)

            # only add axis 2D
            if len(X_t.shape) == 2:
                X_t = X_t[...,np.newaxis]

            X_t = np.repeat(X_t,self.n_channels,axis=2)


            X[i,:] = X_t

            yt = batch_list[1][i]

            y[i] = yt

        if len(self.classes) == 2: # binary
            y[y==self.classes[0]] = 0
            y[y==self.classes[1]] = 1
        else:
            y = to_categorical(y, num_classes=len(self.classes))

        y = y.astype(np.uint8)

        return X, y

