import os


#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

#os.environ["KERAS_BACKEND"] = "mxnet"

os.environ["TF_KERAS"] = "1"

from modelFactory import getModel
#from data import *
import keras2onnx
import DataLoad 
from keras.callbacks import *
from sklearn.model_selection import train_test_split
import numpy as np

import Show

import sys

import tensorflow as tf
from tensorflow import keras

#set this values
TrainingsData = "Training"
modelName = "Model2"
numberOfChunksFortraining = 6


print(tf.version.VERSION)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

numEpochs = 50

X, Y, ValX, ValY = DataLoad.getPaths(TrainingsData + '\\Train',TrainingsData + '\\Label',numberOfChunksFortraining)


TrainGenerator = DataLoad.load3D_file_generator(X,Y,epochs=numEpochs)
ValGenerator = DataLoad.load3D_file_generator(ValX,ValY,epochs=numEpochs)

model = getModel("unet3D")

model.save('Keras3d.hdf5')




print ('Number of arguments:' + str(len(sys.argv)) + 'arguments.')
print ('Argument List:' + str(sys.argv))



filepath = "Checkpoints\saved-model-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='max')

if(len(sys.argv) >= 5):
    model.load_weights(sys.argv[4])


model.fit(TrainGenerator, validation_data=ValGenerator, steps_per_epoch= len(X), validation_steps=len(ValX), callbacks=[checkpoint], epochs=numEpochs)



model.save(modelName + '.hdf5')

model.save_weights(modelName + '_weights.h5')



onnx_model = keras2onnx.convert_keras(model, model.name,target_opset=8)

import onnx
temp_model_file = modelName + '.onnx'
onnx.save(onnx_model, temp_model_file)