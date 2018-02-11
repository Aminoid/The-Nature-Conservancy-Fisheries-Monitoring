## Contributor:	Samir Jha
## Kaggle: The Nature Conservancy Fisheries Monitoring challenge
## VGG19 Model adapted from VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION - Karen Simonyan & Andrew Zisserman

import os, sys, glob

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import shutil
import numpy as np
import pandas as pd
import datetime
import os, sys, glob

import theano

from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Lambda
from keras.preprocessing.image import ImageDataGenerator
#Can't use Adam in inception..Gets slow!
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.utils import get_file

ORIGINAL_DATAPATH = "./train/"
TRAIN_PATH = "./KTRAIN/"
KTH_TRAIN_PATH = TRAIN_PATH+"train"
VALIDATION_PATH = "./KVALID/"
KTH_VALID_PATH = VALIDATION_PATH+"valid"
TEST_PATH = "./TEST/test_stg1/"

NUMBER_OF_FOLDS = 4 

# Adapting stratifiedkfold. Removed startifiedShuffleSplit logic.
 
def stratifiedKFolds(X, Y, num_folds = 3):
#	sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=2017)
#	sss.get_n_splits(X, Y)
#		break
#	return(train_idx, test_idx)

	skf = StratifiedKFold(n_splits=num_folds, shuffle = True, random_state=2017)
	skf.get_n_splits(X, Y)
	TrainIDx = []
	ValidIDx = []
#	for train_idx, test_idx in sss.split(X, Y):
	for train_idx, test_idx in skf.split(X, Y):
		TrainIDx.append(train_idx)
		ValidIDx.append(test_idx)

	return(TrainIDx, ValidIDx)

def createTrainAndValidationDatasets(datapath):
#Preferable sampling using k-folds over stratifiedShuffleSplit used in Inception. Experimenting with accuracy.
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

	TrainIDx, ValidIDx = stratifiedKFolds(img_names, img_classes, NUMBER_OF_FOLDS)

	for i in range(1,len(TrainIDx)+1):
		makeDirectoryStructure(TRAIN_PATH+"/train"+str(i)+"/")
		makeDirectoryStructure(VALIDATION_PATH+"/valid"+str(i)+"/")	

	fold = 1
	for train_idx in TrainIDx:
		for i in range(len(train_idx)):	
			img_name = img_names[train_idx[i]]
			img_class = img_classes[train_idx[i]]
			shutil.copyfile(ORIGINAL_DATAPATH + img_class + "/" + img_name, KTH_TRAIN_PATH + str(fold)+ "/" +img_class + "/" + img_name)
		fold += 1
		
	fold = 1		
	for valid_idx in ValidIDx:
		for j in range(len(valid_idx)):
			img_name = img_names[valid_idx[j]]
			img_class = img_classes[valid_idx[j]]
			shutil.copyfile(ORIGINAL_DATAPATH + img_class + "/" + img_name, KTH_VALID_PATH+ str(fold)+ "/" + img_class + "/" + img_name)
		fold += 1

def loadTrainAndValidationDatasets(size, fold):
	print("### Creating generators...")
	train_datagen = ImageDataGenerator()
	valid_datagen = ImageDataGenerator()
	classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
	train_generator = train_datagen.flow_from_directory(KTH_TRAIN_PATH+str(fold)+"/", target_size=size, batch_size=1, class_mode='sparse', classes=classes)
	valid_generator = valid_datagen.flow_from_directory(KTH_VALID_PATH+str(fold)+"/", target_size=size, batch_size=1, class_mode='sparse', classes=classes)

	print("### Loading datasets...")
	train_len = len(glob.glob(KTH_TRAIN_PATH+str(fold)+"/"+'*/*.jpg'))
	valid_len = len(glob.glob(KTH_VALID_PATH+str(fold)+"/"+'*/*.jpg'))

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

## Not using Adam..Slows down the training..Use faster SGD..

def createVGG19(input_shape=(512, 512, 3), optimizer=Adam(lr=0.001), weights = 'imagenet'):
	base_model = VGG19(weights=weights, include_top=False, input_shape=input_shape,pooling='avg')
	
	x = base_model.layers[-1].output
##	Using higher dropout..Tried with dropout 0.3... 
	x = Dropout(0.5)(x)
#	x = Dense(512, activation='tanh')(x)	
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dense(8, activation='softmax')(x)

	model = Model(inputs=base_model.input, outputs=x)
	model.compile(optimizer=optimizer, loss="categorical_crossentropy")

	return(base_model, model)

##Assuming fresh run without weights.
train = True

#Train Model#
if(train)

	print("### TRAIN variable is set to true. If you merely need to run the model, download vgg_19_weights from https://github.ncsu.edu/achoudh3/The-Nature-Conservancy-Fisheries-Monitoring")
	print("train variable can be set to false and model can be run using pre-trained weights!")
	print("Proceeding with training..Could take a while!")
	
	createTrainAndValidationDatasets(ORIGINAL_DATAPATH)
	
	size = (512, 512)

#	Other team using faster learning rate..Toggle here for faster training..Initial training with slower lr for higher accuracy..
#	lr = 1e-3 runs very slow..Might run even slower for main.py..Step size can be changed to 1e-2 with a decay...	
	optimizer = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=False)
	base,model = createVGG19(optimizer = optimizer)

	for layer in base.layers:
			layer.trainable = False
		
	model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

	print("### Fitting model iteratively over validation folds..")

#	StratifiedShuffleSplit logic removed!!
#	Adapting k-folds...
	
	for i in range(1,NUMBER_OF_FOLDS+1):
		train_X, train_lbls, valid_X, valid_lbls = loadTrainAndValidationDatasets(size,i)
		train_Y = to_categorical(train_lbls)
		valid_Y = to_categorical(valid_lbls)
# 	Higher batch size giving memory overflow in ARC. Check!!
#	Increased epochs for SGD conversion...Last run with 10 epochs..very low accuracy..
		model.fit(x=train_X, y=train_Y, batch_size=16, epochs=40, verbose=2, validation_data=(valid_X, valid_Y))
		del train_X, train_Y,valid_X, valid_Y


	print("### Retraining the last few layers of Keras VGG19 model...")

#	can start retraining earlier, TODO: check this vs accuracy..	
	for layer in base.layers[20:]:
			layer.trainable = True

	model.compile(optimizer=optimizer,  loss="categorical_crossentropy", metrics=['accuracy'])

	print("### Fitting TL model...")

#	scores = model.evaluate(X, Y, verbose=2)
#	print("%s: %.2f%%" % (model.metrics_names[1], scores[]))
#serialize model to JSON
#	vgg19_json = model.to_json()
#	with open("model.json", "w") as json_file:
#   json_file.write(vgg19_json)
#	model.save_weights("vgg19_weights.h5")
#	print("Saved model to disk")

	for i in range(NUMBER_OF_FOLDS):
		train_X, train_lbls, valid_X, valid_lbls = loadTrainAndValidationDatasets(size,i)
		train_Y = to_categorical(train_lbls)
		valid_Y = to_categorical(valid_lbls)
#		This slows down the training time. Use StratifiedShuffleSplit in main.py instead...
		model.fit(x=train_X, y=train_Y, batch_size=16, epochs=40, verbose=2, validation_data=(valid_X, valid_Y) )
		del train_X, train_Y,valid_X, valid_Y

	model.save_weights("vgg_19_weights_4_15.h5")


##TEST VGG19##
##Make sure vgg_19_weights.h5 is present in arc directory##
else

#load json and create model
#	model_file = open('vgg19_json.json', 'r')
#	vgg19_model_json = model_file.read()
#	model_file.close()
#	vgg19_model = model_from_json(vgg19_model_json)
#load weights into new model
#	vgg19_model.load_weights("vgg19_weights.h5")
#	print("Loaded trained vgg19 model")

	optimizer = SGD(lr=1e-3, decay=1e-4, momentum=0.9, nesterov=False)
	base,model = createVGG19(optimizer = optimizer)
	model.load_weights("vgg_19_weights_4_15.h5")

	test_datagen = ImageDataGenerator()

	test_generator = test_datagen.flow_from_directory(TEST_PATH, target_size=size, batch_size=1)

	test_len = len(glob.glob('./TEST//.jpg'))

	batch_size = 1000
	num_batches = (test_len)/batch_size

	if (test_len)%batch_size != 0:
		num_batches += 1

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

	def do_clip(arr, mx): return np.clip(arr, (1-mx)/7, mx)

	classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']


	subm = do_clip(finalpreds,0.82)

	submission = pd.DataFrame(subm, columns=classes)
	submission.insert(0, 'image', testf) 
	submission.head()
	submission.to_csv('./results/srun2/S2.csv', index=False)
