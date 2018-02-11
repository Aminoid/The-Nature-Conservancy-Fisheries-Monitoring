## Abhinav Choudhury
## Kaggle: The Nature Conservancy Fisheries Monitoring challenge

import os, sys, glob
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
import numpy as np
import pandas as pd
import datetime

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
import cv2
import vgg16bn
from vgg16bn import Vgg16BN
import theano
from keras import backend as K

if len(sys.argv) != 2:
	print("Usage: python inception_ft.py <train/predict>")
	exit(0)

if sys.argv[1] == "train":
	train = True
else:
	train = False

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
	classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
	train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=size, batch_size=1, class_mode='sparse', classes=classes)
	valid_generator = valid_datagen.flow_from_directory(VALIDATION_PATH, target_size=size, batch_size=1, class_mode='sparse', classes=classes)

#	TEST_PATH = "./TEST/"
#	test_datagen = ImageDataGenerator()
#	test_generator = test_datagen.flow_from_directory(TEST_PATH, target_size=size, batch_size=1)

	print("### Loading datasets...")
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
	x = Dropout(0.5)(x)
	x = GlobalAveragePooling2D()(x)
	x = Dense(1024, activation='relu')(x)
	x = Dense(8, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=x)
	model.compile(optimizer=optimizer, loss="categorical_crossentropy")

	return(base_model, model)

def addConvBlock(model, layers, nf, input_shape=None):
        for i in range(layers):
                if i == 0 and input_shape is not None:
                        model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
                else:
                        model.add(ZeroPadding2D((1, 1)))
                model.add(Conv2D(nf, (3, 3), activation='relu'))
        model.add(MaxPooling2D((2, 2), strides=(2,2)))


size = (512, 512)

if train:
	train_X, train_lbls, valid_X, valid_lbls = loadTrainAndValidationDatasets(size)
	train_Y = to_categorical(train_lbls)
	valid_Y = to_categorical(valid_lbls)

	start = datetime.datetime.now()

	#### For InceptionV3, first make the base untrainable and train only the top
	print("### [Phase 1] Creating model...")
	optimizer = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=False)
	base, model = createInceptionV3(optimizer=optimizer)
	print("Done")

	for layer in base.layers:
		layer.trainable = False

	print("### [Phase 1] Compiling model...")
	model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

	print("### [Phase 1] Fitting model...training only top")
	model.fit(x=train_X, y=train_Y, batch_size=64, epochs=10, verbose=2, validation_data=(valid_X, valid_Y) )

	#### Now finetune the InceptionV3 model
	print("### [Phase 2] Finetuning started...")
	### Layer 171: for Inception
	### Layer 20: for VGG16
	for layer in base.layers[171:]:
		layer.trainable = True

	print("### [Phase 2] Recompiling model...")
	model.compile(optimizer=optimizer,  loss="categorical_crossentropy", metrics=['accuracy'])

	print("### [Phase 2] Fitting model...")
	model.fit(x=train_X, y=train_Y, batch_size=64, epochs=10, verbose=2, validation_data=(valid_X, valid_Y) )

	print("### Saving weights...")
	model.save_weights("inception_ft_4_16.h5")
	print("Done. Time taken = " + str(datetime.datetime.now() - start))
else:
	print("### Creating model...")

	optimizer = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=False)
        base, model = createInceptionV3(optimizer=optimizer)

	print("### Loading weights...")
	model.load_weights("inception_ft_4_16.h5")

	### Expects testdata as a numpy array of shape (size, 3, x, x)
	def get_im_cv2(path, target_size=(512, 512)):
	    img = cv2.imread(path)
	    resized = cv2.resize(img, target_size, cv2.INTER_LINEAR)
            return resized

	def read_test_images(path, target_size):
		files = glob.glob(path + "*.jpg")
		names = []
		data = []
		for f in files:
			img_data = get_im_cv2(f, target_size=target_size).transpose((2, 0, 1))
			img_data = img_data.astype('float32')/255
			data.append(img_data)
			names.append(os.path.basename(f))
		return(names, data)

	def predict(model, testdata, testids):
		preds = model.predict(testdata, batch_size = 64)
		res = pd.DataFrame(preds, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
		res.loc[:, 'image'] = pd.Series(testids, index=res.index)
		cols = list(res)
		cols = cols[-1:] + cols[:-1]
		res = res[cols]
		res.to_csv("output.csv", index=False)


	def predict_batches(model, path, target_size, batch_size):
                files = glob.glob(path + "*.jpg")
                preds = []
		names = []
		num_batches = len(files)/batch_size
		print("# Total batches to predict:  " + str(num_batches))

		if len(files)%batch_size != 0:
			num_batches += 1

		for i in range(num_batches):
			print("# Reading Batch no. " + str(i+1))
			file_batch = files[i*batch_size: (i+1)*batch_size]
			data = []
			for f in file_batch:
				img_data = get_im_cv2(f, target_size=target_size).transpose((2, 0, 1))
				img_data = img_data.astype('float32')/255
				data.append(img_data)
				names.append(os.path.basename(f))
			print("# Predicting batch...")
			data = np.array(data)
#			out = model.predict(data, batch_size=batch_size)
			out = model([data, 0])[0]
			preds.append(out)

		print("Shape of out = " + str(out.shape))
		preds = np.concatenate(np.array(preds))
		print("Verification: Preds shape = " + str(preds.shape) + ", names = " + str(len(names)) )
		return(preds, names)

	def write_to_csv(preds, names):
		res = pd.DataFrame(preds, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
		res.loc[:, 'image'] = pd.Series(names, index=res.index)
                cols = list(res)
                cols = cols[-1:] + cols[:-1]
                res = res[cols]
		print("# Writing to CSV...")
                res.to_csv("output.csv", index=False)

	### Get embeddings
	train_X, train_lbls, valid_X, valid_lbls = loadTrainAndValidationDatasets((512, 512))

        ### Drop top and save conversions
	f = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-3].output])

	train_out = []
	valid_out = []
	batch_size = 64
	num_batches = train_X.shape[0]/batch_size
	if train_X.shape[0]%batch_size != 0:
		num_batches += 1
	for i in range(num_batches):
		train_activations = f([train_X[i*batch_size:(i+1)*batch_size], 0])
		train_out.append(train_activations)

        num_batches = valid_X.shape[0]/batch_size
        if valid_X.shape[0]%batch_size != 0:
                num_batches += 1

        for i in range(num_batches):
		valid_activations = f([valid_X[i*batch_size:(i+1)*batch_size], 0])
		valid_out.append(valid_activations)

	train_out = np.concatenate(np.array(train_out))
	train_out = np.concatenate(train_out)
	valid_out = np.concatenate(np.array(valid_out))
	valid_out = np.concatenate(valid_out)

        print("Train_activations shape = " + str(train_out.shape))
        print("Valid_activations shape = " + str(valid_out.shape))

        train_df = pd.DataFrame(train_out)
        train_df.loc[:, 'class'] = pd.Series(train_lbls, index=train_df.index)
        valid_df = pd.DataFrame(valid_out)
        valid_df.loc[:, 'class'] = pd.Series(valid_lbls, index=valid_df.index)
        cols = list(train_df)
        cols = cols[-1:] + cols[:-1]
        train_df = train_df[cols]
        valid_df = valid_df[cols]
        train_df.to_csv("train.csv", index=False)
        valid_df.to_csv("valid.csv", index=False)

	print("### Reading and predicting batches...")
	start = datetime.datetime.now()
	test1_preds, test1_names = predict_batches(f, "./TEST/test_stg1/", target_size=(512, 512), batch_size=64)
	test2_preds, test2_names = predict_batches(f, "./TEST/test_stg2/", target_size=(512, 512), batch_size=64)
	test2_names_changed = ["test_stg2/" + name for name in test2_names]

	allPreds = np.concatenate(np.array([test1_preds, test2_preds]))
	print("AllPreds shape = " + str(allPreds.shape))
	allNames = test1_names + test2_names_changed

#	write_to_csv(allPreds, allNames)
        res = pd.DataFrame(allPreds)
        res.loc[:, 'image'] = pd.Series(allNames, index=res.index)
        cols = list(res)
        cols = cols[-1:] + cols[:-1]
        res = res[cols]
        print("# Writing to CSV...")
        res.to_csv("output.csv", index=False)


	print("Done. Time taken = " + str(datetime.datetime.now() - start))

