#%%

#%% md
# 
#%%
try:
    import keras
except ImportError:
    import pip
    pip.main(['install','keras'])
from keras.models import  Sequential
from keras.layers import Dense
from keras.utils import to_categorical
    
#%%

# to add convloutional layers
from keras.layers.convolutional import Conv2D

# to add pooling layers
from keras.layers.convlolutional import MaxPooling2D

#to flatten data for fully connected layers
from keras.layers import Flatten
#%%
# import data
from keras.datasets import mnist

#load data
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0],28,28,1).astype('float32')

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
#%%
X_train = X_train / 255 # normalize training data
X_test = X_test / 255 # normalize test data

Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

num_classes = Y_test.shape[1] # number of categories
#%%
