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

print(tf.version.VERSION)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

numEpochs = 50

X, Y, ValX, ValY = DataLoad.getPaths('.\\396N_without1\Train','.\\396N_without1\Label',16)

#from isensee2017 import *

TrainGenerator = DataLoad.load3D_file_generator(X,Y,epochs=numEpochs)
ValGenerator = DataLoad.load3D_file_generator(ValX,ValY,epochs=numEpochs)

# data_gen_args = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')
                    
# myGene = trainGenerator(2,'data/membrane/train','image','label',data_gen_args,save_to_dir = None)

#data = DataLoad.load3D()

#model = isensee2017_model()
model = getModel("unet3D")



model.save('Keras3d.hdf5')

modelName = "Model2"

#model.fit_generator(data,steps_per_epoch=3,epochs=1,callbacks=[model_checkpoint])

print ('Number of arguments:' + str(len(sys.argv)) + 'arguments.')
print ('Argument List:' + str(sys.argv))

""" if(len(sys.argv) >= 3):
    #X_high,Y_high = DataLoad.load3D_XY(sys.argv[1], sys.argv[2])

    gen_3D = DataLoad.load3D(sys.argv[1], sys.argv[2], epochs=100,numberOfImages=240)

else:
    #load split high resolution
    X_high,Y_high = DataLoad.load3D_XY('E:\DATA\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man8bit+VS_Calibrated_1220x854x976.mhd',
    'E:\DATA\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\General_otsu_BIN_AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.mhd')


print ('Number of arguments:' + str(len(sys.argv)) + 'arguments.')
print ('Argument List:' + str(sys.argv))

if(len(sys.argv) >= 4):
    modelName = sys.argv[3]



#X_train, X_test, y_train, y_test = train_test_split(X_high, Y_high, test_size=0.9, random_state=1)

# #load split low resilution
# X_low,Y_low = DataLoad.load3D_XY('D:\weinberger\Probe2.mhd','D:\weinberger\Probe2_bin.mhd')
# X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, Y_low, test_size=0.2, random_state=1)

# #merge high and lo
# X_train = np.vstack([X_train, X_train_low])
# X_test = np.vstack([X_test, X_test_low])
# y_train = np.vstack([y_train,y_train_low])
# y_test = np.vstack([y_test,y_test_low]) """

filepath = "Checkpoints\saved-model-{epoch:02d}-{loss:.2f}.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=False, mode='max')

if(len(sys.argv) >= 5):
    model.load_weights(sys.argv[4])

model.load_weights('fibre.hdf5')

def init_layer(layer):
    weights = proper_shape0 = layer.get_weights()
    proper_shape0 = layer.get_weights()[0].shape
    proper_shape1 = layer.get_weights()[1].shape
    weights = [ np.random.normal(loc=0.0, scale=0.1, size=proper_shape0),np.random.normal(loc=0.0, scale=0.1, size=proper_shape1)]
    layer.set_weights(weights)

layer = model.get_layer('conv3d_20')
init_layer(layer)
layer = model.get_layer('conv3d_21')
init_layer(layer)
layer = model.get_layer('conv3d_22')
init_layer(layer)
layer = model.get_layer('conv3d_23')
init_layer(layer)

#model.fit(x=gen_3D , callbacks=[checkpoint],steps_per_epoch=240, epochs=100)


#dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(1)
#dataset_val = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(1)
model.fit(TrainGenerator, validation_data=ValGenerator, steps_per_epoch= len(X)/batchSize, validation_steps=len(ValX)/batchSize, callbacks=[checkpoint], epochs=numEpochs)

""" 
TestResults = model.evaluate(X_test,y_test,batch_size=2)

TestPrediction = model.predict(X_test ,batch_size=2)



index =0
for result in TestPrediction:
    Show.transformSave(result,index,(122,122,122))
    index = index +1

print(TestResults)
 """

model.save(modelName + '.hdf5')

model.save_weights(modelName + '_weights.h5')

#results = model.predict(data)

#for i in range(len(results)):
#    Show.transformSave(results[i])

#testGene = testGenerator("data/membrane/test")
#results = model.predict_generator(testGene,30,verbose=1)
#saveResult("data/membrane/test",results)

onnx_model = keras2onnx.convert_keras(model, model.name,target_opset=8)

import onnx
temp_model_file = modelName + '.onnx'
onnx.save(onnx_model, temp_model_file)