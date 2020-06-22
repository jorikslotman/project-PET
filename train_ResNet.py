from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Conv2D, Dense,MaxPooling2D,UpSampling2D,Flatten,Input
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import PIL
import numpy as np
import glob
from DataGenerator_DLPET import DataGenerator
file_list = glob.glob
input_path = r'C:\Users\CYO-Isala\Documents\DLPET\data\Processed\all'

input_tensor = Input(shape=(100, 100, 1))
resnet_model = ResNet50(include_top=False, weights=None, input_tensor=input_tensor)

x = resnet_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=resnet_model.input, outputs=predictions)

opt = Adam(lr = 0.001)
model.compile(optimizer='Adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# CONSTANT
DG = DataGenerator(r'C:\Users\CYO-Isala\Documents\DLPET\data\Processed\all',shuffle=True,dim=(100,100))
model.fit_generator(DG,epochs=3)

