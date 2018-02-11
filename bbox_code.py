## Abhinav Medhekar
## Kaggle: The Nature Conservancy Fisheries Monitoring challenge

import os, sys, glob
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
import numpy as np
import pandas as pd
import datetime
#import guppy

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
#from guppy import hpy; h=hpy()
#import cv2
#import vgg16bn
#from vgg16bn import Vgg16BN

# Perform a stratified split given a vector of images names
# and a vector of corresponding image classes
def stratifiedSplit(X, Y, test_size=0.13238):
	sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=2017)
	sss.get_n_splits(X, Y)
	for train_idx, test_idx in sss.split(X, Y):
		break
	return(train_idx, test_idx)

ORIGINAL_DATAPATH = "./train/"
TRAIN_PATH = "./TRAIN/"
VALIDATION_PATH = "./VALID/"
TEST_PATH = "./TEST/"
path = "./BBFish/"



def loadTrainAndValidationDatasets(size):
	print("### Creating generators...")
	train_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()
        


	#TRAIN_PATH = "./sample/TRAIN/"
	#VALIDATION_PATH = "./sample/VALID/"
	train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=size, batch_size=1)
	valid_generator = valid_datagen.flow_from_directory(VALIDATION_PATH, target_size=size, batch_size=1)
        


	#TEST_PATH = "./TEST/"
	#test_datagen = ImageDataGenerator()
	#test_generator = test_datagen.flow_from_directory(TEST_PATH, target_size=size, batch_size=1)

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




#Method for getting file names of all validation, training and test files
def get_filenames():
    validf = [] 
    for dirs in sorted(os.listdir("./VALID")):
           listvalid  = os.listdir("./VALID/"+dirs)
           flistvalid  = [ "./VALID/"+ dirs + "/"+ s for s in listvalid]
           validf.extend(flistvalid)
    trainf = []
    for dirs in sorted(os.listdir("./TRAIN")):
           listtrain = os.listdir("./TRAIN/"+dirs)
           flisttrain  = [ "./TRAIN/"+ dirs + "/"+ s for s in listtrain]
           trainf.extend(flisttrain)
    testf = []
    for dirs in sorted(os.listdir("./TEST")):
           listtest = os.listdir("./TEST/"+dirs)
           flisttest  = [ "./TEST/"+ dirs + "/"+ s for s in listtest]
           testf.extend(flisttest)
    return (validf,trainf,testf)
    
size = (512, 512)

#code for parsing json files and retrieving information in the dictionary bbox_loc
bbox_loc = {}
for f in sorted(os.listdir("./BBFish/annos/")):
    with open("./BBFish/annos/" + f) as file_stream:
        imgs = json.load(file_stream)
        for eachimg in imgs:
            if 'annotations' in eachimg.keys() and len(eachimg['annotations'] )> 0:
                key = (eachimg['filename'].split('/'))[-1]
                value = eachimg['annotations'][-1]
                bbox_loc[key] = value

(fullvalf, fulltrainf, fulltestf) = get_filenames()

#Remove the path from file name
trainf = [f.split('/')[-1] for f in fulltrainf]
valf = [f.split('/')[-1] for f in fullvalf]
testf= [f.split('/')[-1] for f in fulltestf]


#Assigning empty boxes to those images for which there are no bounding boxes given.
empty_bbox = {'height': 0., 'width': 0., 'x': 0., 'y': 0.}

for f in trainf:
    if not f in bbox_loc.keys(): bbox_loc[f] = empty_bbox
for f in valf:
    if not f in bbox_loc.keys(): bbox_loc[f] = empty_bbox

bb_params = ['height', 'width', 'x', 'y']


#Scale (X, Y, Height, Width) for bounding box to required size i.e 512*512. 
def scale(bb, size):
    bb = [bb[p] for p in bb_params]
    conv_x = (512. / size[0])
    conv_y = (512. / size[1])
    bb[0] = bb[0]*conv_y
    bb[1] = bb[1]*conv_x
    bb[2] = max(bb[2]*conv_x, 0)
    bb[3] = max(bb[3]*conv_y, 0)
    return bb

#list of scaled bounding box values
train_box_list = []
for f in fulltrainf:
   size = PIL.Image.open(f).size
   bb_scaled = scale(bbox_loc[f.split('/')[-1]], size)
   train_box_list.append(bb_scaled)
 
#list of scaled bounding box values
valid_box_list = []
for f in fullvalf:
   size = PIL.Image.open(f).size
   bb_scaled = scale(bbox_loc[f.split('/')[-1]], size)
   valid_box_list.append(bb_scaled)

#Creating list to arrays    
train_box = np.array(train_box_list).astype(np.float32)
val_box = np.array(valid_box_list).astype(np.float32)

#Loading training and validation sets
train_X, train_Y, valid_X, valid_Y = loadTrainAndValidationDatasets(size)

input_shape=(512, 512,3)

#Load pre-computed model.
model = load_model('bb_reg.h5')


#Split the precomputed model at the last convolutional layer.
intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer('conv2d_94').output)


#Get the predictions at this level for training and validation data
intermediate_output = intermediate_layer_model.predict(train_X)

intermediate_val_output = intermediate_layer_model.predict(valid_X)

#New model comprising 2 more dense layers and then 2 separate output layers.
p=0.6

inp = Input(intermediate_layer_model.get_layer('conv2d_94').output_shape[1:])
x = MaxPooling2D()(inp)
x = BatchNormalization()(x)
x = Dropout(p/4)(x)
x = Flatten()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p)(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(p/2)(x)
x_bb = Dense(4, name='bb')(x)
x_class = Dense(8, activation='softmax', name='class')(x)

model = Model([inp], [x_bb, x_class])

model.compile(Adam(lr= 0.001), loss=['mse', 'categorical_crossentropy'], metrics=['accuracy'], loss_weights=[.001, 1.])

model.optimizer.lr = (1e-5)

model.fit(intermediate_output, [train_box, train_Y], batch_size=32, nb_epoch=10, 
             validation_data=(intermediate_val_output, [val_box, valid_Y]))


#Code for reading in the test data
test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_directory(TEST_PATH, target_size=size, batch_size=1)

test_len = len(glob.glob('./TEST/*/*.jpg'))

#We use a batch size of 1000 as we cannot put the whole 13000 images into a numpy at the same time(memory issues)
batch_size = 1000
num_batches = (test_len)/batch_size

if (test_len)%batch_size != 0:
	num_batches += 1

#In batches of 1000, feed the testing data to the network and predict its output. Store this output in finalpreds.
finalpreds = []
for i in range(num_batches):
       xytuples = []
       minnum = min(test_len,((i+1)*batch_size))
       for j in range(i*batch_size , minnum):
                x = test_generator.next()
                xytuples.append(x[0])
       test_X = np.concatenate([x for x in xytuples])
       test_X = test_X/ 255
       intermediate_output = intermediate_layer_model.predict(test_X)
       preds = model.predict(intermediate_output) 
       finalpreds.extend(preds[1])


classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

#Code for getting the results in CSV format.
submission = pd.DataFrame(subm, columns=classes)
submission.insert(0, 'image', testf) 
submission.head()
submission.to_csv('./results/srun2/S2.csv', index=False) 


##################References#################################################
# https://github.com/fastai/courses/blob/master/deeplearning1/nbs/lesson7.ipynb
# https://keras.io/applications/


