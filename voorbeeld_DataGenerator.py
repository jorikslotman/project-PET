import os
import glob
import cv2
import tempfile
import random as rn
import math
import time

import numpy as np
import pandas as pd

from scipy.misc import imread
from vis.utils import utils as visutils
from scipy.ndimage import gaussian_filter
from skimage.segmentation import clear_border
from scipy.ndimage.measurements import center_of_mass
from skimage.feature import peak_local_max

import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.metrics import auc
from tensorflow.keras.models import load_model
from tensorflow import local_variables_initializer
from tensorflow.keras.utils import Sequence, to_categorical

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import auc as skauc

class DataGenerator(Sequence):

    # folder with classes separated in sub directories
    # only for binary so far! (no categorial)

    def __init__(
            self, 
            folder, 
            batch_size = 32, 
            dim = (512,512), 
            n_channels = 1,
            shuffle = True,
            correct_SI = False,
            bg_mask = False,
            noise = None,
            hist_eq = False,
            augment = False,
            augment_params = dict(),
            label_noise = None
            ):

        self.dim = dim
        self.batch_size = batch_size
        self.folder = folder
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.correct_SI = correct_SI
        self.bg_mask = bg_mask
        self.hist_eq = hist_eq
        self.augment = augment
        self.noise = noise
        self.label_noise = label_noise

        if self.augment:
            imgen = ImageDataGenerator(**augment_params)
            self.random_transform = imgen.random_transform 

        file_list = []
        class_list = []

        for d in glob.glob(os.path.join(self.folder,'*','')):
            file_list_t = glob.glob(os.path.join(d,'*.png'))
            file_list.extend(file_list_t)
            class_list.extend([os.path.basename(os.path.dirname(d))] * len(file_list_t))

        self.data_list = np.array([file_list,class_list])
        self.classes = np.unique(self.data_list[1,:])

        self.on_epoch_end()


    def __len__(self):

        return int(np.ceil(self.data_list.shape[1] / self.batch_size))


    def __getitem__(self,index):

        if index > np.floor(self.data_list.shape[1] / self.batch_size): # incomplete batch
            batch_indexes = self.indexes[index * self.batch_size:] # grab the last ones from the list
        else:
            batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        batch_list = self.data_list[:,batch_indexes]
        self.batch_list = batch_list

        X, y = self.__data_generation(batch_list)

        return X, y

    def histogram_equalization(self,image, range_intensities=4000):
        image_histogram, bins = np.histogram(image.flatten(), range_intensities, density=True)
        cdf = range_intensities * image_histogram.cumsum() / image_histogram.cumsum()[-1]
        image = np.reshape(np.interp(image.flatten(), bins[:-1], cdf), image.shape)
        return image

    def on_epoch_end(self):

        self.indexes = np.arange(self.data_list.shape[1])
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self,batch_list):

        X = np.empty((batch_list.shape[1], *self.dim,self.n_channels),dtype='float')
        y = np.empty((batch_list.shape[1]), dtype='|U16')

        for i,file_t in enumerate(batch_list[0,:]):

            X_t = np.array(imread(file_t,flatten=True))

            if self.correct_SI == True:
                ii = file_t.rfind('_I')
                I = int(file_t[ii+2:ii+6]) # intersect
                si = file_t.rfind('_S')
                S = int(file_t[si+2:si+6]) # slope
                X_t = X_t/S
                X_t -= I

            # normalize
            X_t = X_t.astype(np.float)
            pu = np.percentile(X_t,1)

            # X_t -= pu
            X_t -= -1050 # lower boundary is set to -1050 instead of 1%
            pa = np.percentile(X_t, 99)
            X_t = X_t/pa

            # resize to self.dim
            X_t = cv2.resize(X_t, dsize=self.dim, interpolation=cv2.INTER_CUBIC)
            X_t = np.clip(X_t, 0, 1)

            # remove background
            if self.bg_mask == True:
                X_tt = X_t*255
                ret2, th2 = cv2.threshold(X_tt.astype(np.uint8), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                th2i = 1 - th2 # get inverse from background mask
                th2i2 = clear_border(th2i) # find non-border connected regions in inversed mask
                mask = th2+th2i2 # add non-border region to mask
                mask = np.clip(mask,0,1) # ensure binary
                self.mask = mask

            # histogram equalization
            if self.hist_eq == True:
                X_t = self.histogram_equalization(X_t*4000,range_intensities=4000)
                X_t /= 4000

            # apply mask
            if self.bg_mask == True:

                X_t[mask==0] = 0

                if self.noise=='uniform':
                    masku = np.random.uniform(low=0.0, high=1.0, size=self.dim) * (1 - mask)
                    X_t += masku
                if self.noise == 'gaussian_smoothed':
                    maskg = gaussian_filter(np.random.uniform(low=0.0, high=1.0, size=self.dim),3)*(1 - mask)
                    X_t += maskg
                if self.noise == 'grad_gaussian_smoothed':
                    maskg = gaussian_filter(mask.astype(np.float), 100)
                    masku = np.random.uniform(low=-1.0, high=1.0, size=self.dim)
                    maskf = masku + maskg
                    mask_grad = np.clip(gaussian_filter(maskf,3),0,1)
                    mask_grad *= (1-mask)

                    X_t += mask_grad

            # only add axis 2D
            if len(X_t.shape) == 2:
                X_t = X_t[...,np.newaxis]

            X_t = np.repeat(X_t,self.n_channels,axis=2)

            # apply augmentation
            if self.augment:
                X_t = self.random_transform(X_t) 

            X[i,:] = X_t

            yt = batch_list[1][i]

            # apply label noise (swap classes with given probability) - only works for binary!
            if self.label_noise != None:
                if rn.random() < self.label_noise:
                    yt = self.classes[np.where(self.classes != yt)[0][0]]

            y[i] = yt

        if len(self.classes) == 2: # binary
            y[y==self.classes[0]] = 0
            y[y==self.classes[1]] = 1
        else:
            y = to_categorical(y, num_classes=len(self.classes))

        y = y.astype(np.uint8)

        return X, y

class Analysis():
    def __init__(self,file_path):
        self.file_path = file_path
        self.df = pd.read_csv(self.file_path)

    def maxAcc(self):
        print('Max val_acc: ' +str(np.round(self.df['val_acc'].max(),3)) + ' at epoch ' + str(self.df['epoch'][self.df['val_acc'].idxmax()]) + '.')

    def graph(self,col_ind = [1]):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.xlabel(self.df.columns.values[0])
        X = self.df.iloc[:,0].values
        legend_list = []
        for c in col_ind:
            Y = self.df.iloc[:,c].values
            plt.plot(X,Y)
            legend_list.append(self.df.columns.values[c])
        plt.legend(legend_list)
        plt.show()

class saliency_mapper():
    def __init__(self,model):
        # compile saliency
        inp = model.layers[0].input
        outp = model.layers[-1].output
        max_outp = K.max(outp, axis=1)
        saliency = K.gradients(K.sum(max_outp), inp)[0]
        max_class = K.argmax(outp, axis=1)

        self.F = K.function([inp], [saliency, max_class]) # saliency function

    def get_map(self,X,smooth = 5):
        saliency = self.F([X])[0]
        s = visutils.normalize(np.abs(saliency))
        s_m = np.mean(s, axis=3)
        saliency_map = np.zeros(s_m.shape)
        for i in range(s_m.shape[0]):
            saliency_map[i,...] = gaussian_filter(s_m[i,...], sigma=smooth)

        return saliency_map

def auc_djs(y_true, y_pred):
    auc_djs = auc(y_true, y_pred)[1]
    K.get_session().run(local_variables_initializer())
    return auc_djs

def binary_focal_loss(gamma=2., alpha=.25):
   """
   Binary form of focal loss.
     FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
     where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
   References:
       https://arxiv.org/pdf/1708.02002.pdf
   Usage:
    model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
   """
   def binary_focal_loss_fixed(y_true, y_pred):
       """
       :param y_true: A tensor of the same shape as `y_pred`
       :param y_pred:  A tensor resulting from a sigmoid
       :return: Output tensor.
       """
       pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
       pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

       epsilon2 = K.epsilon()
       # clip to prevent NaN's and Inf's
       pt_1 = K.clip(pt_1, epsilon2, 1. - epsilon2)
       pt_0 = K.clip(pt_0, epsilon2, 1. - epsilon2)

       return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
              -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

   return binary_focal_loss_fixed

def apply_modifications(model, custom_objects=None):
    """Applies modifications to the model layers to create a new Graph. For example, simply changing
    `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
    with modified inbound and outbound tensors because of change in layer building function.
    Args:
        model: The `keras.models.Model` instance.
    Returns:
        The modified model with changes applied. Does not mutate the original `model`.
    """
    # The strategy is to save the modified model and load it back. This is done because setting the activation
    # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
    # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
    # multiple inbound and outbound nodes are allowed with the Graph API.
    model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model.save(model_path)
        return load_model(model_path, custom_objects=custom_objects)
    finally:
        os.remove(model_path)

def AUC(y_true,y_pred):
    def optimal_t(fpr, tpr, thresholds):
        optimal_idx = np.argmax(tpr - fpr)
        optimal_t = thresholds[optimal_idx]
        return optimal_t

    un,co = np.unique(y_pred,return_counts=True)
    un = np.sort(un)
    TPR = []
    FPR = []
    c = 0
    for t in un:
        c += 1
        p = y_pred>=t
        TPR.append(np.sum(y_true[p])/np.sum(y_true))
        FPR.append(np.sum(1-y_true[p]) / np.sum(1-y_true))

    from matplotlib.pyplot import plot,title
    plot(FPR,TPR)
    a = np.round(skauc(FPR,TPR),4)
    print('AUC: ' + str(a))
    title('AUC: ' + str(a))
    return a

def top_x_len(s_map,y_map,x=5):
    # s = 2D sal_map
    # y_map = 2D label_map
    # x = top x lengths
    loc_max = peak_local_max(s_map, min_distance=10)
    intensities = s_map[loc_max[:,0],loc_max[:,1]]
    sort_order = np.argsort(intensities)[::-1]
    tx_loc_max = loc_max[sort_order[:x]]

    # all separate areas in label map
    areas = np.unique(y_map)[np.unique(y_map) > 0]
    num_areas = int(np.shape(areas)[0])

    # pre-allocate
    tot_distances = np.zeros([x,num_areas])

    # iterate over separate ares
    for i,a in enumerate(areas):
        y_map_a = np.zeros(y_map.shape)
        y_map_a[y_map==a] = 1
        y_c_mass = np.round(center_of_mass(y_map_a))
        tx_distances = np.array(list(map(lambda t : math.sqrt((tx_loc_max[t,1]-y_c_mass[1])**2+(tx_loc_max[t,0]-y_c_mass[0])**2),range(tx_loc_max.shape[0]))))
        tot_distances[:,i] = tx_distances

    # take minimum distance
    distances = np.min(tot_distances,axis=1)

    return tx_loc_max, distances