import os


#os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

#os.environ["KERAS_BACKEND"] = "mxnet"

from modelFactory import getModel
from data import *
import keras2onnx
import DataLoad 
from keras.callbacks.callbacks import *
from sklearn.model_selection import train_test_split

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

X,Y = DataLoad.load3D_XY()

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)




model.fit(X_train, y_train,batch_size=1,epochs=20,validation_split=0.2)


TestResults = model.evaluate(X_test,y_test,batch_size=2)

TestPrediction = model.predict(X_test ,batch_size=2)

index =0
for result in TestPrediction:
    Show.transformSave(result,index,(122,122,122))
    index = index +1

model.save('Keras3d.hdf5')

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