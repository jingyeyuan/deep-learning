# deep-learning
computer vision (face recognition)


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import keras.backend as K


K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt


import keras.backend as K
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt

def mean_pred(y_true,y_pred):
    return K.mean(y_pred)

def load_dataset():
    train_dataset = h5py.File('train_happy.h5','r')
    train_set_x_orig =  np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    
    test_dataset = h5py.File('test_happy.h5','r')
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])
    
    classes =  np.array(test_dataset["list_classes"][:])
    
    train_set_y_orig = train_set_y_orig.reshape((1,train_set_x_orig.shape[0])) #convert the dataset to (1,shape of train_y)
    test_set_y_orig = test_set_y_orig.reshape((1,test_set_y_orig.shape[0])) #convert the dataset to (1, shape of test_y)
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
    
    
    X_train_orig,Y_train_orig,X_test_orig,Y_test_orig,classes=load_dataset()

#normalize
X_train = X_train_orig/225
X_test = X_test_orig/225

#reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

def HappyModel(input_shape):
    x_input=Input(input_shape)
    x=ZeroPadding2D((3,3))(x_input)
    x=Conv2D(32,(7,7),strides=(1,1),name='conv0')(x)
    x=BatchNormalization(axis=3,name='bn0')(x)
    x=Activation('relu')(x)
    
    x=MaxPooling2D((2,2),name='max_pool')(x) #maxpooling之后的层的大小为（input size/size of maxpooling）
    x=Flatten()(x)
    x=Dense(1,activation='sigmoid',name='fc')(x) #fully connect, 1是output的unit个数 Dense(units, activation)
    
    model=Model(inputs=x_input,outputs=x,name='HappyModel') #Model(inputs=a,outputs=b)模型包含从a到b的计算的所有网络层
    
    return model
                                        
happymodel = HappyModel((64,64,3))
happymodel.compile(optimizer='adam',loss='binary_crossentropy',metrics=["accuracy"])
#model类模型的方法compile(optimizer, loss, metrics);optimizer,loss,metrics都是字符串，metrics一般选用['accuracy']
happymodel.fit(x=X_train,y=Y_train,epochs=40,batch_size=16) #training

preds = happymodel.evaluate(x=X_test, y=Y_test) #testing and predicting
print()
print("loss = "+str(preds[0])) #preds的第一个为loss
print("accuracy ="+str(preds[1])) #preds的第二个位accuracy

img_path='test_picture.png'
img=image.load_img(img_path,target_size=(64,64))
x=image.img_to_array(img)
x=np.expand_dims(x,axis=0)
x=preprocess_input(x)
print(happymodel.predict(x))
