import os


#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

#os.environ["KERAS_BACKEND"] = "mxnet"

from modelFactory import getModel
from data import *
import keras2onnx
import DataLoad 
from keras.callbacks.callbacks import *
from sklearn.model_selection import train_test_split
import numpy as np

import Show

#from isensee2017 import *



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
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)

model.save('Keras3d.hdf5')



#model.fit_generator(data,steps_per_epoch=3,epochs=1,callbacks=[model_checkpoint])


#load split high resolution
X_high,Y_high = DataLoad.load3D_XY('E:\DATA\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.mhd',
'E:\DATA\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\General_otsu_BIN_AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.mhd')
X_train, X_test, y_train, y_test = train_test_split(X_high, Y_high, test_size=0.2, random_state=1)

# #load split low resilution
# X_low,Y_low = DataLoad.load3D_XY('E:\weinberger\Probe2.mhd','E:\weinberger\Probe2_bin.mhd')
# X_train_low, X_test_low, y_train_low, y_test_low = train_test_split(X_low, Y_low, test_size=0.2, random_state=1)

# #merge high and low
# X_train = np.vstack([X_train, X_train_low])
# X_test = np.vstack([X_test, X_test_low])
# y_train = np.vstack([y_train,y_train_low])
# y_test = np.vstack([y_test,y_test_low])

model.fit(X_train, y_train,batch_size=1,epochs=10,validation_split=0.2)


TestResults = model.evaluate(X_test,y_test,batch_size=2)

TestPrediction = model.predict(X_test ,batch_size=2)

index =0
for result in TestPrediction:
    Show.transformSave(result,index,(122,122,122))
    index = index +1

model.save('Keras3d.hdf5')

print(TestResults)

#results = model.predict(data)

#for i in range(len(results)):
#    Show.transformSave(results[i])

#testGene = testGenerator("data/membrane/test")
#results = model.predict_generator(testGene,30,verbose=1)
#saveResult("data/membrane/test",results)

onnx_model = keras2onnx.convert_keras(model, model.name,target_opset=8)

import onnx
temp_model_file = 'model.onnx'
onnx.save(onnx_model, temp_model_file)