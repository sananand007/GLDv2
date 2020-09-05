# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import re
import json
import sys
import cv2
import os
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

TRAIN_ROOT_PATH='../input/landmark-recognition-2020/train/'
TEST_ROOT_PATH='../input/landmark-recognition-2020/test/'
SIZE = 256

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        #print(os.path.join(dirname, filename))
        finalPath = os.path.join(dirname, filename)
        if 'train.csv' in filename:
            dfTrain = pd.read_csv(finalPath)
# print(dfTest.head(5))
print(dfTrain.columns)
print(dfTrain.head(5))

#checking Samples and classes in the training dataset
'''
sort_values(by, axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
'''

def _plot_samples(data, field):
    fil_df = data.groupby([field]).count().sort_values(by=['id'], ascending=False)
    print("Get the 1st 20")
    print(fil_df[:20])
    print("Get the last 20")
    print(fil_df[:-20])
    return fil_df

# Group by landmark_id, 
clTr_df=_plot_samples(dfTrain, field='landmark_id')

#check the number of classes present
def number_of_classes(data):
    print(f'The Number of classes/landmark id that are present here are {len(data)}')

number_of_classes(clTr_df)


#Look at the images and the sizes
def read_image(path, im_size, normalize_range=False):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (im_size, im_size))
    print(f'The shape of the image {img.shape}')

#check some sample images
def check_images(data, low, high):
    ids = data['id'][low:high]
    for _id in ids:
        name = _id+'.jpg'
        path = TRAIN_ROOT_PATH+_id[0]+'/'+_id[1]+'/'+_id[2]+'/'+name
        print(f'The image path is {path}')
        read_image(path=path, im_size=SIZE)

check_images(data=dfTrain, low=0, high=2)

######################################################################################################################
'''
Base Model to Test Dataset
'''

base = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="../input/tutorial-keras-transfer-learning-with-resnet50/best.hdf5",
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classifier_activation="softmax"
        )

inputs = keras.Input(shape=(224, 224, 3))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()
x = keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(N_CLASSES, activation='softmax')(x)
model = keras.Model(inputs, out)

model.summry()
