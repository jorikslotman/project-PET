from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense,MaxPooling2D,UpSampling2D,Flatten
import PIL
import numpy as np
import glob
from DataGenerator_DLPET import DataGenerator
file_list = glob.glob
input_path = r'C:\Users\CYO-Isala\Documents\DLPET\data\Processed\all'
file_list_X = glob.glob(input_path+'\\half\\*')[:10]
file_list_Y = glob.glob(input_path+'\\full\\*')[:10]


X = np.zeros([10,288,288])
Y = np.zeros([10,288,288])

for i in np.arange(10):
    Xt = np.array(PIL.Image.open(file_list_X[i])).astype(np.float)
    Yt = np.array(PIL.Image.open(file_list_Y[i])).astype(np.float)

    X[i,...] = Xt
    Y[i,...] = Yt

X = X[...,np.newaxis]
Y = Y[...,np.newaxis]

model = Sequential()

model.add(Conv2D(6,(3,3),input_shape=(288,288,1), activation='relu',padding='same'))
model.add(Conv2D(6,(3,3),input_shape=(288,288,1), activation='relu',strides=(2,2),padding='same'))
model.add(Conv2D(1,(3,3),input_shape=(288,288,1), activation='relu',strides=(2,2),padding='same'))
model.add(Flatten())
model.add(Dense(1,  activation='sigmoid'))
model.summary()


model.compile(optimizer='Adam',
              loss='mean_absolute_error',
              metrics=['mean_absolute_error'])
# CONSTANT
DG = DataGenerator(r'C:\Users\CYO-Isala\Documents\DLPET\data\Processed\all',shuffle=True)
model.fit_generator(DG,epochs=3)

