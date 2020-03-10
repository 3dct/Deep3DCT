from keras.models import load_model
import DataLoad 
import Show
from metrics import dice_coef_loss

#load split high resolution
X,Y = DataLoad.load3D_XY('E:\DATA\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.mhd',
'E:\DATA\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\General_otsu_BIN_AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.mhd')



model = load_model('Keras3d.hdf5', custom_objects={'dice_coefficient_loss':                   
dice_coef_loss})

TestResults = model.evaluate(X,Y,batch_size=2)

#TestPrediction = model.predict(X_test ,batch_size=2)

# index =0
# for result in TestPrediction:
#     Show.transformSave(result,index,(122,122,122))
#     index = index +1


print(TestResults)