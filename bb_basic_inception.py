## Abhinav Medhekar
## Kaggle: The Nature Conservancy Fisheries Monitoring challenge

import os, sys, glob
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
import numpy as np
import pandas as pd
import datetime

from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, Adagrad, SGD, RMSprop
from keras.utils import to_categorical
from keras.models import Model, load_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.utils import get_file
import ujson as json
from keras.preprocessing import image, sequence
import PIL
from PIL import Image
import bcolz

ORIGINAL_DATAPATH = "./train/"
TRAIN_PATH = "./TRAIN/"
VALIDATION_PATH = "./VALID/"
TEST_PATH = "./TEST/"
path = "./BBFish/"

def loadTrainAndValidationDatasets(size):
	print("### Creating generators...")
	train_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=size, batch_size=1)
	valid_generator = valid_datagen.flow_from_directory(VALIDATION_PATH, target_size=size, batch_size=1)
        


	print("### Creating datasets...")
	train_len = len(glob.glob('./TRAIN/*/*.jpg'))
	valid_len = len(glob.glob('./VALID/*/*.jpg'))
     


	xytuples = []
	for i in range(train_len):
	        x = train_generator.next()
	        xytuples.append(x)

	train_X = np.concatenate([x[0] for x in xytuples])
	train_Y = np.concatenate([y[1] for y in xytuples])

	xytuples = []
	for i in range(valid_len):
	        x = valid_generator.next()
	        xytuples.append(x)

	valid_X = np.concatenate([x[0] for x in xytuples])
	valid_Y = np.concatenate([y[1] for y in xytuples])

       	print("Train X shape = " + str(train_X.shape))
	print("Train Y shape = " + str(train_Y.shape))
	train_X = train_X / 255
	valid_X = valid_X / 255
       

	return(train_X, train_Y, valid_X, valid_Y)

size = (512, 512)

#Load trainig and validation data
train_X, train_Y, valid_X, valid_Y = loadTrainAndValidationDatasets(size)

input_shape=(512, 512,3)

#Initialize the base inception model.
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)

#Add dense and 8-softmax layer on top for classifying the 8 fish categories
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)

model = Model(input=base_model.input, output=predictions)

#Now make base model untrainable and only train the layers added on top.
for layer in base_model.layers:
    layer.trainable = False

#Compile and fit this model
model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=['accuracy'])

model.fit(x=train_X, y=train_Y, batch_size=32, epochs=10, verbose=2, validation_data=(valid_X, valid_Y) )


#Now for Transfer learning, we try to fine tune the base Inception V3 model to our needs. We disable the botton 171 layers and train the remaining ones on the top.
# The parameter 171 was used in the standard Keras documentation at the site https://keras.io/applications/. We tried keeping the same and got good results,
# so we decided to continue with the same.
for layer in base_model.layers[171:]:
    layer.trainable = True

#Now compile and fit the fine tuned model.
model.compile(optimizer=SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=False), loss='categorical_crossentropy', metrics=['accuracy'])
        
model.fit(x=train_X, y=train_Y, batch_size=32, epochs=10, verbose=2, validation_data=(valid_X, valid_Y) )

#Save the model for further use.
model.save('bb_reg.h5')


########################## References#####################################################################
# Code from standard Keras documentation for Inception V3: https://keras.io/applications/ has been used here. We have made modifications
# to the parameters for achieving the best possible results. 
