#import glob
from imutils import paths
import os
import cv2
import keras.preprocessing.image
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

dataset_path=(r"C:\Users\Meghna Prabhu\Desktop\ML\ML_proj\data")


imgpaths=list(paths.list_images(dataset_path))
X=[]
Y=[]
        #loop over the input image
for(i, imgpath) in enumerate(imgpaths):
    
    img=cv2.imread(imgpath)
    label=(os.path.basename(imgpath).split('_'))[2]
    X.append(img)
    Y.append(label)
    
X = np.asarray(X)
Y = np.asarray(Y)
    
print(X.shape)
X = X.reshape(X.shape[0], 28, 28, 1)
X = X.astype('float32')
X /= 255



train_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)

# Generates batches of Augmented Image data
train_generator = train_datagen.flow_from_directory('train/', target_size = (300, 300),batch_size = batch_size,class_mode = 'binary') 

# Generator for validation data
validation_generator = test_datagen.flow_from_directory('test/', target_size = (300, 300),   batch_size = batch_size, class_mode = 'binary')