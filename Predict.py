from keras.models import load_model
import DataLoad 
import Show
from metrics import dice_coef_loss

#load split high resolution
X,Y = DataLoad.load3D_XY('\\\\10.49.10.81\CT-dev\PWeinberger\General_Otsu_AI_Referenz_CFK_3_3um_Probe3_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\AI_Referenz_CFK_3_3um_Probe3_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.mhd',
'\\\\10.49.10.81\CT-dev\PWeinberger\General_Otsu_AI_Referenz_CFK_3_3um_Probe3_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\Multiple_Otsu_AI_Referenz_CFK_3_3um_Probe3_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.mhd')



model = load_model('C:\\Users\Patrick\Downloads\PWeinberger\OnlyHigh\Keras3d.hdf5', custom_objects={'dice_coefficient_loss':                   
dice_coef_loss})

import keras2onnx
onnx_model = keras2onnx.convert_keras(model, model.name,target_opset=8)

import onnx
temp_model_file = 'model_dice.onnx'
onnx.save(onnx_model, temp_model_file)

TestResults = model.evaluate(X,Y,batch_size=2)

# TestPrediction = model.predict(X ,batch_size=2)

# index =0
# for result in TestPrediction:
#     Show.transformSave(result,index,(122,122,122),name='Test1/Predicted_')
#     index = index +1

# index =0
# for result in X:
#     Show.transformSave(result,index,(132,132,132),name='Test1/Image_')
#     index = index +1

# index =0
# for result in Y:
#     Show.transformSave(result,index,(122,122,122), name='Test1/Reference_')
#     index = index +1


# print(TestResults)