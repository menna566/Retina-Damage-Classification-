import tensorflow as tf
import numpy as np
import cv2
import os

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import Dense
from keras.layers import Dropout,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
import pickle
from glob import glob
import skimage
from sklearn.utils import shuffle
from keras.optimizers import adam_v2
import seaborn as sns
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, Input
from tensorflow.keras.callbacks import EarlyStopping, LambdaCallback
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
test_ratio=0.2
val_ratio=0.2
noOfclasses = 4
################################################################################
def read_train():
    image_size=(128,128,3)
    path1 ="/kaggle/input/kermany2018/OCT2017 /train"
    myList = os.listdir(path1)
    x_train=[]
    y_train=[]
    CATEGORIES = ['NORMAL',"CNV","DME","DRUSEN"]
    for x in myList:
      myPicList = os.listdir(path1+"/"+str(x))
      for y in myPicList[:10]:
        curImg = cv2.imread(path1+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(image_size[0],image_size[1]))
        x_train.append(curImg)
        y_train.append(CATEGORIES.index(x))
    x_train = np.array(x_train)
    y_train = np.array(y_train)    
    return x_train,y_train
##################################################################################
def get_test():
    image_size=(128,128,3)
    path2 ="/kaggle/input/kermany2018/OCT2017 /test"
    x_test=[]
    y_test=[]
    myList = os.listdir(path2)
    CATEGORIES = ['NORMAL',"CNV","DME","DRUSEN"]
    for x in myList:
      myPicList = os.listdir(path2+"/"+str(x))
      for y in myPicList[:10]:
        curImg = cv2.imread(path2+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(image_size[0],image_size[1]))
        x_test.append(curImg)
        y_test.append(CATEGORIES.index(x))
    x_test = np.array(x_test)
    y_test = np.array(y_test)  
    return x_test,y_test
#################################################################################
def get_val():
    image_size=(128,128,3)
    path3 ="/kaggle/input/kermany2018/OCT2017 /val"
    x_val=[]
    y_val=[]
    myList = os.listdir(path3)
    CATEGORIES = ['NORMAL',"CNV","DME","DRUSEN"]
    for x in myList:
      myPicList2 = os.listdir(path3+"/"+str(x))
      for y in myPicList2[:5]:
        curImg = cv2.imread(path3+"/"+str(x)+"/"+y)
        curImg = cv2.resize(curImg,(image_size[0],image_size[1]))
        x_val.append(curImg)
        y_val.append(CATEGORIES.index(x))
    x_val = np.array(x_val)
    y_val = np.array(y_val)  
    return x_val,y_val

def shapes():
    print(x_train.shape)
    print(x_test.shape)
    print(x_val.shape)

    print(y_train.shape)
    print(y_test.shape)
    print(y_val.shape)
############################################################################
def under_sampling():
    image_size = (128,128,3)
    X_trainShape = x_train.shape[1]*x_train.shape[2]*x_train.shape[3]
    X_trainFlat = x_train.reshape(x_train.shape[0], X_trainShape)
    Y_train = y_train
    ros = RandomUnderSampler()
    X_trainRos, Y_trainRos = ros.fit_resample(X_trainFlat, Y_train)
    # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    Y_trainRosHot = to_categorical(Y_trainRos, num_classes = 4)
    # Make Data 2D again
    for i in range(len(X_trainRos)):
        height, width, channels = image_size[0],image_size[1],3
        X_trainRosReshaped = X_trainRos.reshape(len(X_trainRos),height,width,channels)
    return X_trainRosReshaped,Y_trainRosHot,Y_trainRos

################################################################################
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
################################################################################
def reshaping(X_trainRosReshaped,x_test,x_val,y_test,y_val,y_train):
        x_train = np.array(list(map(preProcessing,X_trainRosReshaped)))
        x_test = np.array(list(map(preProcessing,x_test)))
        x_validation = np.array(list(map(preProcessing,x_val)))

        X_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
        X_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)
        X_validation = x_validation.reshape(x_validation.shape[0],x_validation.shape[1],x_validation.shape[2],1)

        y_train = to_categorical(Y_trainRos,noOfclasses)
        y_test = to_categorical(y_test,noOfclasses)
        y_validation = to_categorical(y_val,noOfclasses)
        return X_train,y_train,X_test,y_test,X_validation,y_validation  
#################################################################################
def model():
  filters=60
  sizeoffilter1 = (5,5)
  sizeoffilter2 = (4,4)
  sizeoffilter3 = (3,3)
  sizeofpool = (2,2)
  node=5000

  model = Sequential();
  model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(128,128,1),activation='relu',kernel_initializer='he_uniform',))
    
  model.add((Conv2D(filters,sizeoffilter1,activation="relu")))
  model.add((Conv2D(filters//2,sizeoffilter2,activation="relu")))
  model.add((Conv2D(filters//2,sizeoffilter2,activation="relu")))
  model.add(MaxPooling2D(pool_size=sizeofpool))
  model.add(Dropout(0.2))

  model.add((Conv2D(filters,sizeoffilter1,activation="relu")))
  model.add((Conv2D(filters//2,sizeoffilter2,activation="relu")))
  model.add((Conv2D(filters//2,sizeoffilter2,activation="relu")))
  model.add(MaxPooling2D(pool_size=sizeofpool))
  model.add(Dropout(0.2))

  model.add((Conv2D(filters,sizeoffilter2,activation="relu")))
  model.add(MaxPooling2D(pool_size=sizeofpool))
  model.add((Conv2D(filters//2,sizeoffilter3,activation="relu")))
  model.add((Conv2D(filters//2,sizeoffilter3,activation="relu")))
  model.add(MaxPooling2D(pool_size=sizeofpool))
  model.add(Dropout(0.2))

  model.add(Flatten())
  model.add(Dense(node,activation="relu"))
  model.add(Dropout(0.2))
  model.add(Dense(noOfclasses,activation="softmax"))
  return model
#################################################################################################
def run_model():
        model.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])
        #model.summary()
        history = model.fit(X_train,Y_trainRosHot,epochs=50,validation_data =(X_validation,y_validation) ,batch_size=256,
                            shuffle=True,
                            max_queue_size=20,
                            use_multiprocessing=True,
                            workers=1)
#########################################################################################################

x_train,y_train = read_train()
x_test,y_test = get_test()
x_val,y_val = get_val()
#shapes()
X_trainRosReshaped,Y_trainRosHot,Y_trainRos = under_sampling()
X_train,y_train,X_test,y_test,X_validation,y_validation = reshaping(X_trainRosReshaped,x_test,x_val,y_test,y_val,y_train)   
model=model()
run_model()