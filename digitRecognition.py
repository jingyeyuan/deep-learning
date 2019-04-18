import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.utils.np_utils import to_categorical  # convert numbers to one-hot-encoding
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model

import keras.backend as K

K.set_image_data_format('channels_last')

import matplotlib.pyplot as plt

import pandas as pd
train=pd.read_csv('digit-recognizer/train.csv')
test=pd.read_csv('digit-recognizer/test.csv')
Y_train_orig=train['label']
X_train_orig=train.drop(labels='label',axis=1)
Y_train_orig=Y_train_orig.values.reshape((Y_train_orig.shape[0]))


from sklearn.model_selection import train_test_split
X_train_orig,X_val,Y_train_orig,Y_val = train_test_split(X_train_orig,Y_train_orig,test_size=0.1,random_state=2)


#Normalization
X_train_orig=X_train_orig.values.reshape(-1,28,28,1)
test=test.values.reshape(-1,28,28,1)
X_train=X_train_orig/225
X_test=test/255

#reshape
Y_train=to_categorical(Y_train_orig,num_classes=10)

Y_val=to_categorical(Y_val,num_classes=10)
X_val=X_val.values.reshape(-1,28,28,1)
X_val=X_val/225

#model
def digitRecognition(input_shape):
    x_input = Input(input_shape)
    x = ZeroPadding2D((3, 3))(x_input)
    x = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(x)
    x = BatchNormalization(axis=3, name='bn0')(x)
    x = Activation('relu')(x)
    
    x = MaxPooling2D((2, 2), name='max_pool')(x)
    x = Flatten()(x)
    x = Dense(10, activation='softmax', name='fc')(x)
    
    model = Model(input=x_input, output=x, name='digitRecognition')
    return model

digitR=digitRecognition((28,28,1))
digitR.fit(x=X_train,y=Y_train,epochs=42,batch_size=200)#training epochs and batch_size can be adjusted to optimize
preds=digitR.evaluate(x=X_val,y=Y_val)
print()
print("loss="+str(preds[0]))
print("accuracy="+str(preds[1]))

#test
Y_test=digitR.predict(X_test)
Y_test=np.argmax(Y_test,axis=1)

Y_ans=pd.Series(Y_test,name='Label')

ans=pd.concat([pd.Series(range(1,Y_ans.shape[0]+1),name='ImageId'),Y_ans],axis=1)
print(ans[:10])


submission=ans.to_csv('submission.csv')


