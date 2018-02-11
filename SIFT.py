# Author = Amit Kanwar (akanwar2)

# For running this, you will need to build opencv2 from the source as
# normal installation doesn't include the SIFT APIs.
# I recommend looking at https://ianlondon.github.io/blog/how-to-sift-opencv/ 
# I would like to say thanks to LearnDeltaX Tutorial on SIFT with KNN and K-Means 
# for helping me with the implementation.

# The train, test_stg1 and test_stg2 directories should be in the same directory as 
# SIFT.py for this to run



import os
import argparse
import glob
import cv2
import numpy as np
from scipy.cluster import vq

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

CLASSES = {
    'ALB': 1,
    'BET': 2,
    'DOL': 3,
    'LAG': 4,
    'NoF': 5,
    'OTHER': 6,
    'SHARK': 7,
    'YFT': 8
}

CLASSES_REV = {value: key for key, value in CLASSES.items()}

def _load_img(path):
    img = cv2.imread(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def petty_print(msg):
    print('=' * len(msg))
    print(msg)
    print('=' * len(msg))


def _detect_and_describe(image):
    descriptor = cv2.xfeatures2d.SIFT_create()
    (kps, features) = descriptor.detectAndCompute(image, None)
    kps = np.float32([kp.pt for kp in kps])
    features = np.float32(features)
    return kps, features


def _kmeans_clustering(data, k=7):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    ret, label, centers = cv2.kmeans(data, k, None, criteria, 10, flags)
    return centers

def extract_img_features(img_data_dir, type):
    if type == 'train':
        files = glob.glob("%s/*/*" % img_data_dir)
    else:
        files = glob.glob("%s/*" % img_data_dir)
        files += glob.glob("%s/*" %('./test_stg2'))
    dataset_size = len(files)
    resp = np.zeros((dataset_size, 1))
    ctr = 0
    print("\nCreating descriptors from Images\n")
    des_list = []
    dpt_list = []
    #des_list = np.empty((0))
    for f in files:
        print("Describing image %s" % f)
        img = _load_img(f)
        kpts, des = _detect_and_describe(img)
        des_list.append(des)
        dpt_list.append(des[0])
        if type == 'train':
            resp[ctr] = CLASSES[f.split('/')[-2]]
            ctr += 1
    descriptors = np.asarray(dpt_list)
    #petty_print("\nStacking the descriptors..\n")
    #for image_path, descriptor in des_list[1:]:
    #    descriptors = np.vstack((descriptors, descriptor))

    print("\nK-means Clustering..\n")
    centers = _kmeans_clustering(descriptors, 7)
    print len(des_list[0])
    print len(centers[0])
    im_features = np.zeros((dataset_size, 7), "float32")
    for i in range(dataset_size):
        words, distance = vq.vq(des_list[i], centers)
        for w in words:
            im_features[i][w] += 1

    stdSlr = StandardScaler().fit(im_features)
    im_features = stdSlr.transform(im_features)

    resp = np.float32(resp)
    return files, im_features, resp


def train_classifier(train_data, train_resp):
    model = KNeighborsClassifier(weights='distance', n_jobs=-1)
    model.fit(train_data, train_resp)
    return model


def test_classifier(model, test_data):
    result = model.predict_proba(test_data)
    return result


if __name__ == '__main__':
    training_data_dir = './train'
    testing_data_dir = './test_stg1'

    petty_print("Extracting training image features")
    train_files, train_data, train_resp = extract_img_features(training_data_dir, 'train')
    petty_print("Training the classifier")
    model = train_classifier(train_data, train_resp)

    if os.path.exists(testing_data_dir):
        petty_print("Extracting testing image features")
        test_files, test_data, test_resp = extract_img_features(testing_data_dir, 'test')

        petty_print("Testing the classifier")
        predictions = test_classifier(model, test_data)
        columns = [CLASSES_REV[int(entry)] for entry in model.classes_]
        submission1 = pd.DataFrame(predictions, columns=columns)
        images = []
        for f in test_files:
            x = f.split('/')
            if x[1] == 'test_stg2':
                images.append('/'.join(x[-2:]))
            else:
                images.append(x[-1])
        submission1.insert(0, 'image', images)
        submission1.head()
        submission1.to_csv("final.csv", index=False)
