######################################################################  Atit Shetty (akshetty)###################################################################
from sklearn.model_selection import StratifiedKFold
import shutil
import numpy as np
import datetime
import os, sys, glob

import theano

from keras import backend as K
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.utils import get_file

ORIGINAL_DATAPATH = "./train/"
KTH_TRAIN_PATH = "./KTRAIN/train"
KTH_VALID_PATH = "./KVALID/valid"
TEST_PATH = "./TEST/test_stg1/"

NUMBER_OF_FOLDS = 4 

###Generated K-folds based on Class distribution Y
def stratifiedKFolds(X, Y, num_folds = 3):
	skf = StratifiedKFold(n_splits=num_folds, shuffle = True, random_state=2017)
	skf.get_n_splits(X, Y)
	
	TrainIDx = []
	ValidIDx = []

	for train_idx, test_idx in skf.split(X, Y):
		TrainIDx.append(train_idx)
		ValidIDx.append(test_idx)

	return(TrainIDx, ValidIDx)

###Save the records in directory
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

	TrainIDx, ValidIDx = stratifiedKFolds(img_names, img_classes, NUMBER_OF_FOLDS)

	for i in range(1,len(TrainIDx)+1):
		makeDirectoryStructure(KTH_TRAIN_PATH+str(i)+"/")
		makeDirectoryStructure(KTH_VALID_PATH+"/valid"+str(i)+"/")	

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

###Create and return VGG16 FCN model
def createVGG16(input_shape=(512, 512, 3), optimizer=Adam(lr=0.001), weights = 'imagenet'):
	base_model = VGG16(weights=weights, include_top=False, input_shape=input_shape)
	
	x = base_model.layers[-1].output
	x = Dropout(0.5)(x)
	x = Conv2D(256,(3,3),activation='relu')(x)
	x = Conv2D(8,(3,3),activation='relu')(x)
	x = GlobalAveragePooling2D()(x)
	x = Activation('softmax')(x)

	model = Model(inputs=base_model.input, outputs=x)
	model.compile(optimizer=optimizer, loss="categorical_crossentropy")

	return(base_model, model)

###set True to train VGG16 and create weights file
train = False 

##########################################################TRAIN VGG16##############################################################################
if train:
	
	###Run once to generate directory
	#createTrainAndValidationDatasets(ORIGINAL_DATAPATH)

	size = (512, 512)

	optimizer = SGD(lr=1e-3, decay=1e-4, momentum=0.89, nesterov=False)

	base,model = createVGG16(optimizer = optimizer)
	#model.summary()
	for i in range(1,NUMBER_OF_FOLDS+1):
		if i > 1:
			model.load_weights("vgg_16_fcn_kfolds.h5")

		for layer in base.layers:
				layer.trainable = False

		model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])

		print("### Model Fitting")

		
		train_X, train_lbls, valid_X, valid_lbls = loadTrainAndValidationDatasets(size,i)
		train_Y = to_categorical(train_lbls)
		valid_Y = to_categorical(valid_lbls)
		model.fit(x=train_X, y=train_Y, batch_size=32, epochs=4, verbose=2, validation_data=(valid_X, valid_Y))

		print("### Model Finetuning Phase")

		for layer in base.layers[16:]:
				layer.trainable = True

		model.compile(optimizer=optimizer,  loss="categorical_crossentropy", metrics=['accuracy'])

		print("### Fitting model...")

		model.fit(x=train_X, y=train_Y, batch_size=32, epochs=4, verbose=2, validation_data=(valid_X, valid_Y) )
		del train_X, train_Y,valid_X, valid_Y

		print("### Saving weights...")
		model.save_weights("vgg_16_fcn_kfolds.h5")


############################################################TEST VGG16################################################################################
else:
	optimizer = SGD(lr=1e-2, decay=1e-4, momentum=0.89, nesterov=False)
	base,model = createVGG16(optimizer = optimizer)
	model.load_weights("vgg_16_fcn_kfolds_atit.h5")


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
