# The-Nature-Conservancy-Fisheries-Monitoring
Kaggle Competition

## Files
* install-requirements.sh -- Install necessary packages and dependencies
* inception_ft.py		-- Main InceptionV3 CNN implementation file
* lgb.py			-- GBDT implementation using [Microsoft LightGBM](https://github.com/Microsoft/LightGBM)
* SIFT.py			-- SIFT Implementation file

## Compute environment setup
__NOTE__: This experiment will not run in a normal laptop/PC or will take a long time.
We suggest a system with a powerful GPU for tensor processing.

Our setup: ARC compute node **c74/c76** with Nvidia GTX TitanX GPU, 3072 cores, 1000 MHz core clock, 12 GB memory

Setup.sh script provides code for getting most required packages set up on an ARC node.

## How to run
	python inception_ft.py train	# Train the InceptionV3 model
	python inception_ft.py predict	# Generate embeddings for train and test images
	python lgb.py

## Details
![Image](https://github.ncsu.edu/achoudh3/The-Nature-Conservancy-Fisheries-Monitoring/blob/master/Model.png)

We use a Keras InceptionV3 model with weights trained on Imagenet.
On top, we add the following layers to generate a 8-class probability vector corresponding to 8 classes in our dataset.

	Dropout(0.5)
	GlobalAveragePooling2D()
	Dense(1024, 'relu')	--- 1
	Dense(8, 'softmax')	--- 2
	
First only the added top layers are trained by making InceptionV3 layers untrainable.
Next, the InceptionV3 model is finetuned to our dataset by training only layers with indices 171 onwards. This essentially keeps the top half of the Inception model as it is.
Now, remove the added top Dense layers (1 and 2)
Output of CNN is now a 2048-vector. This will let us generate embeddings for our images.

Generate embeddings of training and test dataset and output to CSV files.

### LightGBM for fast gradient boosting trees
Use LightGBM to fit gradient boosted trees on the training set embeddings.
Use the best cross validated model for test set probabilities.

1. model = lgb.LGBMClassifier(boosting_type='gbdt', objective='multiclass', num_leaves=60, max_depth=5, learning_rate=0.01, n_estimators=200, subsample=1, colsample_bytree=0.8, reg_lambda=0)

## Results
Public leaderboard log-loss: **1.28766** with above settings
All models with higher n_estimators seem to overfit resulting in worse log-loss scores

## Test data
The original Kaggle test data (based on which the final leaderboard scores were submitted) can be found here as test_stg1.zip and test_stg2.zip:
https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring/data


## Method 2: Using Bounding Box Regression

Files added:
bbox_code.py
bb_basic_inception.py

Folder:
BBFish(contains the json file containing bounding box locations)

Description and Results:
The Inception V3 used as a base layer here is the same as the one used for the Method 1.
Specifically, we take the output of the conv2d_94 layer of the fine tuned model, add a couple of dense layers on top and then give out 2 separate outputs. One for the bounding boxes and one for the actual classes.

This model gives a very high validation accuracy on our validation data set(>97%), however it does badly on the test data set provided by Kaggle and gives a logloss of 2.4. We believe that this is due to overfitting on the training data and different types of boats used in the test set. 

Steps for running:
First run bb_basic_inception.py

This should generate bb_reg.h5

After this go ahead and run bbox_code.py. Remember this code requires the presence of test data in a folder called TEST_PATH = "./TEST/". The link for this has been shared.This also requires a folder called BBFish which has the JSON files for bounding box locations.


## CNN Model 2: 

### VGG19 model:
Files added:
dVGG19.py

Notes: The Visual Geometry Groups's CNN model(Paper: ICLR 2015) retrained for Fish Species classification and couple of dense layers added on top. Various hyperparameters tried for optimizations and training. Gains expected with more epochs with better SGD convergence and by ensembling with LGB..
	
Directions to run: 
```
python dVGG19.py #Train the model
```
Once trained and the weight file is created, turn train flag to False and re-run.
```
python dVGG19.py #Loads model with weight file and runs prediction on test dataset
```

## CNN Model 3:
Files added:
```
vgg16_fcn_kfolds.py and vgg16_fcn_kfolds.h5
```
Directions to run:
```
To train, set train flag as true.
Change weights file name and run the script.

To test, set train flag as false.
Run the script.
```
## CNN Model 4:
Files added:
```
resnet.py 
```
Directions to run:
```
python resnet.py ##train model

To test, set train flag as false.

Run the script.

```

## Method 3: Using SIFT
Files added:
```
SIFT.py
```
Directions to run:
```
Before running this, you will need to build opencv from scratch. 
I recommend following the approach mentioned here: 
https://ianlondon.github.io/blog/how-to-sift-opencv/.

After setting the opencv environment, just run the script. 
