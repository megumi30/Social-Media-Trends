import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread_collection
from skimage.transform import rescale, resize, downscale_local_mean

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras import backend as K
import tensorflow.keras.layers
import keras
from tensorflow.keras.preprocessing.text import Tokenizer, text_to_word_sequence

#Defining the input image size - chosen so it can be recreated easily
input_img = Input(shape=(320, 192, 3))

#Model architecture definition
#Consists of convultional layers to help learn the features of the clothing item - sleeve type, striped, collar type, etc
x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_img, decoded)

optimizer = keras.optimizers.Adam(lr=0.001)
autoencoder.compile(optimizer=optimizer, loss='binary_crossentropy')



col_dir = 'C:/Users/ACER/Desktop/Flipkart Images Laptop-20220730T102636Z-001/Flipkart Images Laptop/*.jpg'
col_fk = imread_collection(col_dir)
col_fk = list(col_fk)

for i in range(len(col_fk)):
    col_fk[i] = resize(col_fk[i], (320, 192, 3))

col_laptops = col_fk

x_train = np.array(col_laptops[0:300])
x_test = np.array(col_laptops[300:])
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Training the model
autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=32,
                validation_data=(x_test, x_test))

#Saving the model
encoder = Model(input_img, encoded)
encoder.save("encoding")