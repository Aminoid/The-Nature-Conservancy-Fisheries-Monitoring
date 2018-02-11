## Abhinav Choudhury
## Kaggle: The Nature Conservancy Fisheries Monitoring challenge

import os, sys, glob
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, Adagrad, SGD, RMSprop
from keras.utils import to_categorical
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.utils import get_file

import vgg16bn
from vgg16bn import Vgg16BN

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

def createTrainAndValidationDatasets(datapath):
	print("### Sampling training and validation datasets...")
	classes = ["LAG", "DOL", "OTHER", "BET", "ALB", "NoF", "YFT", "SHARK"]

	def makeDirectoryStructure(path):
		if not os.path.exists(path):
			for cl in classes:
				os.makedirs(path + cl)
		else:
			shutil.rmtree(path)
			for cl in classes:
				os.makedirs(path + cl)

	img_names = []
	img_classes = []
	for cl in classes:
		class_dir = ORIGINAL_DATAPATH + cl + "/"
		filepaths = glob.glob(class_dir + "*.jpg")
		for filepath in filepaths:
			img_names.append(os.path.basename(filepath))
			img_classes.append(cl)

	train_idx, valid_idx = stratifiedSplit(img_names, img_classes)
	makeDirectoryStructure(TRAIN_PATH)
	makeDirectoryStructure(VALIDATION_PATH)

	for idx in train_idx:
		img_name = img_names[idx]
		img_class = img_classes[idx]
		shutil.copyfile(ORIGINAL_DATAPATH + img_class + "/" + img_name, TRAIN_PATH + img_class + "/" + img_name)

	for idx in valid_idx:
		img_name = img_names[idx]
		img_class = img_classes[idx]
		shutil.copyfile(ORIGINAL_DATAPATH + img_class + "/" + img_name, VALIDATION_PATH + img_class + "/" + img_name)

#createTrainAndValidationDatasets(ORIGINAL_DATAPATH)


def loadTrainAndValidationDatasets(size):
	print("### Creating generators...")
	train_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()

	#TRAIN_PATH = "./sample/TRAIN/"
	#VALIDATION_PATH = "./sample/VALID/"
	train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=size, batch_size=1)
	valid_generator = valid_datagen.flow_from_directory(VALIDATION_PATH, target_size=size, batch_size=1)

	TEST_PATH = "./TEST/"
	test_datagen = ImageDataGenerator()
	test_generator = test_datagen.flow_from_directory(TEST_PATH, target_size=size, batch_size=1)

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



# Define, create and return a basic convolutional Keras model
def createModel(input_shape=(512, 512, 3)):
	model = Sequential()
	
	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
	model.add(Conv2D(32, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))

	model.add(Flatten())
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(8, activation='softmax'))

	return(model)

vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1))
def vgg_preprocess(x):
    x = x - vgg_mean
    return(x)
    return x[:, ::-1] # reverse axis rgb->bgr
#	return(x)

def addFCBlock(model):
	model.add(Dense(4096, activation='relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.5))

def addConvBlock(model, layers, nf, input_shape=None):
	for i in range(layers):
		if i == 0 and input_shape is not None:
			model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
		else:
			model.add(ZeroPadding2D((1, 1)))
		model.add(Conv2D(nf, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2, 2), strides=(2,2)))

def createFCNmodel(input_shape=(3, 512, 512)):
	model = Sequential()

#	model.add(Lambda(vgg_preprocess, input_shape=(3, 224, 224), output_shape=(3, 224, 224)))

#	model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
#	model.add(MaxPooling2D((2, 2), strides=(2, 2)))


	addConvBlock(model, 2, 64, input_shape=input_shape)
	addConvBlock(model, 2, 128)
	addConvBlock(model, 3, 256)
	addConvBlock(model, 3, 512)
	addConvBlock(model, 3, 512)	
	model.add(Flatten())
	addFCBlock(model)
	addFCBlock(model)
	model.add(Dense(1000, activation='softmax'))

	fname = 'vgg16_bn.h5'
	FILE_PATH="http://www.platform.ai/models/"
        model.load_weights(get_file(fname, FILE_PATH+fname, cache_subdir='models'))

	model.pop()
	for layer in model.layers:
		layer.trainable = False
	model.add(Dense(8, activation='softmax'))

	return(model)

	nf = 128
	p = 0.5

	layers = [
			BatchNormalization(axis=1, input_shape=model.layers[-1].output_shape[1:]),
			Conv2D(nf, (3,3), activation='relu', padding='same'),
			BatchNormalization(axis=1),
			MaxPooling2D(),
			Conv2D(nf, (3,3), activation='relu', padding='same'),
			BatchNormalization(axis=1),
       			MaxPooling2D(),
       			Conv2D(nf, (3,3), activation='relu', padding='same'),
     			BatchNormalization(axis=1),
        		MaxPooling2D(),
        		Conv2D(8, (3,3), padding='same'),
        		Dropout(p),
        		GlobalAveragePooling2D(),
        		Activation('softmax')
		 ]
	
#	for layer in layers:
#		model.add(layer)

	for layer in model.layers:
		print(layer.name)
		print(layer.input_shape)
		print(layer.output_shape)
		print("==============")

	return(model)


### Create InceptionV3 model for transfer learning
### NOTE that loading InceptionV3 model with Theano currently throws an error
### which can be fixed by making the following change:
### Open /home/achoudh3/.local/lib/python2.7/site-packages/keras/engine/topology.py
### In function preprocess_weights_for_loading function, add the line at the start:
### weights = np.array(weights)

### NOTE: Revert this change once done, since this causes issues with VGG16 loading

### Note input_first for Theano input_dim_ordering
def createInceptionV3(input_shape=(3, 512, 512), optimizer=Adam(lr=0.001)):
	base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
	
	x = base_model.layers[-1].output
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
#	x = Dense(2048, activation='relu')(x)
	x = Dense(8, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=x)
	model.compile(optimizer=optimizer, loss="categorical_crossentropy")

	return(base_model, model)



def createResNetModel(input_shape=(512, 512, 3), optimizer=Adam(lr=0.001)):
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

        x = base_model.layers[-1].output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
	x = Dense(2048, activation='relu')(x)
        x = Dense(8, activation='softmax')(x)

        model = Model(input=base_model.input, outputs=x)
        model.compile(optimizer=optimizer, loss="categorical_crossentropy")

        return(base_model, model)


### Model taken from 
### https://www.kaggle.com/cetusparibus/the-nature-conservancy-fisheries-monitoring/fish-keras-forked

def create_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224), dim_ordering='th'))
    model.add(Conv2D(8, (3, 3), activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(Dropout(0.2))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Conv2D(16, (3, 3), activation='relu', dim_ordering='th', init='he_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(96, activation='relu',init='he_uniform'))
    model.add(Dropout(0.4))
    model.add(Dense(24, activation='relu',init='he_uniform'))
    model.add(Dropout(0.2))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=False)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model



size = (512, 512)
train_X, train_Y, valid_X, valid_Y = loadTrainAndValidationDatasets(size)



#### For InceptionV3, first make the base untrainable and train only the top
print("### Creating model...")
optimizer = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=False)
base, model = createInceptionV3(optimizer=optimizer)
print("Done")

for layer in base.layers:
	layer.trainable = False

print("### Compiling model...")
model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=['accuracy'])

print("### Fitting model...")
model.fit(x=train_X, y=train_Y, batch_size=32, epochs=10, verbose=2, validation_data=(valid_X, valid_Y) )


#### Not finetune the InceptionV3 model
print("### Finetune phase started...")
for layer in base.layers[171:]:
	layer.trainable = True

print("### Recompiling model...")
model.compile(optimizer=optimizer,  loss="categorical_crossentropy", metrics=['accuracy'])

print("### Fitting model...")
model.fit(x=train_X, y=train_Y, batch_size=32, epochs=10, verbose=2, validation_data=(valid_X, valid_Y) )

print("### Saving weights...")
model.save_weights("inception_ft.h5")

#print("### Predicting...")
# Need to work on prediction code
#valid_generator = ImageDataGenerator()
#valid_generator = valid_datagen.flow_from_directory(VALIDATION_PATH, target_size=(512, 512), batch_size=20)

#print(model.predict_generator(test_generator, steps=55))
