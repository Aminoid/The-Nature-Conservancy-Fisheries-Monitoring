## Sudipto Biswas
import os, sys, glob
import shutil
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.layers.core import Dropout, Flatten, Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, Adadelta, Adagrad, SGD, RMSprop
from keras.layers import Dense, Activation
from keras.models import Model
from keras.utils import to_categorical
from keras.applications.resnet50 import ResNet50
from keras.utils import get_file
from keras import backend as K
8
def stratifiedSplit(X, Y, test_size=0.1186):
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=13)
    sss.get_n_splits(X, Y)
    for train_idx, test_idx in sss.split(X, Y):
        break
    return (train_idx, test_idx)

ORIGINAL_DATAPATH = "./train/"
TRAIN_PATH = "./TRAIN/"
VALIDATION_PATH = "./VALID/"
TEST_PATH = "./TEST/"

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

def loadTrainAndValidationDatasets(size):
    print("### Creating generators...")
    train_datagen = ImageDataGenerator()
    valid_datagen = ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(TRAIN_PATH, target_size=size, batch_size=1)
    valid_generator = valid_datagen.flow_from_directory(VALIDATION_PATH, target_size=size, batch_size=1)
    TEST_PATH = "./TEST/"
    test_datagen = ImageDataGenerator()
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
    return (train_X, train_Y, valid_X, valid_Y)

def createResNETMod(input_shape=(512, 512, 3), optimizer=Adam(lr=0.001)):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    x = base_model.layers[-1].output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(2048, activation='relu')(x)
    x = Dense(8, activation='softmax')(x)
    model = Model(input=base_model.input, outputs=x)
    model.compile(optimizer=optimizer, loss="categorical_crossentropy")
    return (base_model, model)

## Create weights file
NUMBER_OF_FOLDS = 4
train = True
size = (512, 512)
##TRAIN ResNET
if train:
    optimizer = SGD(lr=1e-5, decay=1e-7, momentum=0.77, nesterov=False)
    base, model = createResNETMod(optimizer=optimizer)
    # model.summary()
    for i in range(1, NUMBER_OF_FOLDS + 1):
        if i > 1:
            model.load_weights("ResNETmod.h5")
        for layer in base.layers:
            layer.trainable = False
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
        print("### Model Fitting")
        train_X, train_lbls, valid_X, valid_lbls = loadTrainAndValidationDatasets(size, i)
        train_Y = to_categorical(train_lbls)
        valid_Y = to_categorical(valid_lbls)
        model.fit(x=train_X, y=train_Y, batch_size=32, epochs=4, verbose=2, validation_data=(valid_X, valid_Y))
        print("### Model Finetuning Phase")
        for layer in base.layers[16:]:
            layer.trainable = True
        model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=['accuracy'])
        print("### Fitting model...")
        model.fit(x=train_X, y=train_Y, batch_size=32, epochs=4, verbose=2, validation_data=(valid_X, valid_Y))
        del train_X, train_Y, valid_X, valid_Y
        print("### Saving weights...")
        model.save_weights("ResNETmod.h5")

##TEST ResNET
else:
    optimizer = SGD(lr=1e-5, decay=1e-6, momentum=0.77, nesterov=False)
    base, model = createResNETMod(optimizer=optimizer)
    model.load_weights("ResNETmod_test.h5")
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow_from_directory(TEST_PATH, target_size=size, batch_size=1)
    test_len = len(glob.glob('./TEST//.jpg'))
    batch_size = 1000
    num_batches = (test_len) / batch_size
    if (test_len) % batch_size != 0:
        num_batches += 1
    finalpreds = []
    for i in range(num_batches):
        xytuples = []
        minnum = min(test_len, ((i + 1) * batch_size))
        for j in range(i * batch_size, minnum):
            x = test_generator.next()
            xytuples.append(x[0])
        test_X = np.concatenate([x for x in xytuples])
        test_X = test_X / 255
        intermediate_output = intermediate_layer_model.predict(test_X)
        preds = model.predict(intermediate_output)
        finalpreds.extend(preds[1])