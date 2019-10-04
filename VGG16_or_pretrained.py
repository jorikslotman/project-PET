import numpy as np
import pandas as pd
from tensorflow.keras.layers import GaussianNoise,concatenate, Input, Flatten,Dense,GlobalAveragePooling2D,Activation,Conv2D,MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1,l2,l1_l2
from tensorflow.keras.applications import VGG16
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint, CSVLogger


import os
import glob
import pickle
from time import time

from DJS_DL import AUC,vis_layer,acc,DataGenerator,auc_djs

'''
Original VGG16 pretrained. No combination with size information.
insert in image_dir wanted image location (with or without scaled images)
'''
datadir = 'D:\\Studie\\M2.3 Stage TG\\Publication\\data\\processed\\clean_3cm_299px\\'
image_dir = datadir + 'images_or_size\\'
input_shape = (299, 299)
K = pickle.load(open(datadir + "pickle\\K.pickle", 'rb'))

augment_params = {
    'rotation_range': 45,
    'width_shift_range': 0.1,
    'height_shift_range': 0.1,
    'shear_range': 0.75,
    'zoom_range': [.75, 1.25],
    'horizontal_flip': True
}

params = {
    'batch_size': 32,
    'dim': input_shape,
    'n_channels': 3,
    'shuffle': True,
    'return_size': False,
    'rem_bg': 'smooth_noise',
    'hist_eq': True,
    'augment_params': augment_params
}
size_file = datadir + 'ORIGINAL_PROCESSED_size_2.xlsx'

vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(input_shape) + (3,))

# Make sure that the pre-trained bottom layers are not trainable
for layer in vgg_model.layers:  # until 15 for last three conv. layers trainable
    layer.trainable = False

# Getting output tensor of the last VGG layer that we want to include
x = vgg_model.output

# Stacking a new simple convolutional network on top of it
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
x = GaussianNoise(0.3)(x)
x = Dense(128, activation='relu')(x)  # dense layer 2
x = Dropout(0.2)(x)
x = GaussianNoise(0.3)(x)
x = Dense(1, activation='sigmoid')(x)  # final layer with softmax activation

# Creating new model. Please note that this is NOT a Sequential() model.
model = Model(inputs=vgg_model.input, outputs=x)

# add regularizers
# to all convolutional layers
for layer in model.layers:
    if 'Dense' in str(layer):
        model.get_layer(layer.name).kernel_regularizer = l1(0.002)

n = 8
SA = {}
models = {}
accs = []
ra = range(3)
# 84 epochs
for rep in ra:
    model.load_weights('reset_weights.h5')
    ar = np.arange(3)
    ar = np.delete(ar,rep)

    training_gen = DataGenerator(np.append(K[ar[0]],K[ar[1]]),image_dir,size_file,label_noise=0.05,augment=True,**params)
    validation_gen = DataGenerator(K[rep],image_dir,size_file,label_noise=None,augment=False,**params)



    adam = optimizers.Adam(lr=0.0001, beta_1=0.5, beta_2=0.75, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy',auc_djs])

    G = glob.glob('logs\\pretrained_or_rep'+str(rep)+'_'+str(n)+'(*).log')
    csv_logger = CSVLogger('logs\\pretrained_or_rep'+str(rep)+'_'+str(n)+'('+str(len(G))+').log')

    start_time = time()
    model.fit_generator(
        generator=training_gen,
        validation_data=validation_gen,
        callbacks=[csv_logger],
        use_multiprocessing=False,
        workers=6,
        verbose=2,
        epochs=84
    )


    elapsed_time = time()-start_time

    val_params = {
        'batch_size' : 15,
        'dim' : input_shape,
        'n_channels': 3,
        'shuffle':False,
        'return_size' : False,
        'rem_bg' : 'smooth_noise',
        'hist_eq' : True,
        'augment': False,
    }

    validation_gen = DataGenerator(K[rep],image_dir,size_file,**val_params)

    y_p = model.predict_generator(validation_gen) # categorial output. First column is zero, second column 1
    y_p = np.squeeze(y_p,axis=1)
    y_t = np.array(list(map(lambda x : validation_gen.numerize_classes[x[-5]],validation_gen.patient_list)))

    y_p_acc = y_p>0.5
    y_p_acc = y_p_acc.astype('uint8')

    AUC(y_t,y_p,plot=True)

    print('Right: ' + str(np.sum(y_p_acc==y_t)))
    print('Wrong: ' + str(np.sum(y_p_acc!=y_t)))
    print(str(np.round(np.sum(y_p_acc==y_t)/len(y_t), 3) * 100) + ' %')
    accs.append(np.round(np.sum(y_p_acc == y_t) / len(y_t), 3) * 100)

    models['model_'+str(rep)] = model
    SA['K_'+str(rep)] = validation_gen.patient_list
    SA['yp_'+str(rep)] = y_p
    SA['yt_'+str(rep)] = y_t

print('Done.')
print(accs)

'''
writer = pd.ExcelWriter('results\\results_final.xlsx')
for rep in range(3):
    SAt = {}
    SAt['K_'+str(rep)] = SA['K_'+str(rep)]
    SAt['yp_'+str(rep)] = SA['yp_'+str(rep)]
    SAt['yt_'+str(rep)] = SA['yt_'+str(rep)]
    df = pd.DataFrame(SAt)

    df.to_excel(writer, str(rep))
writer.save()
for rep in range(3):
    models['model_'+str(rep)].save('final\\final_model_'+str(rep)+'.h5')

'''