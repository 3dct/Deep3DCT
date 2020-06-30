from modelFactory import HyperparameterTune
from metrics import dice_coef_loss
import DataLoad 
from sklearn.model_selection import train_test_split
import talos

X,Y = DataLoad.load3D_XY('D:\DATA\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.mhd',
'D:\DATA\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\General_otsu_BIN_AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.mhd')

#X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

gen_3D = DataLoad.load3D(sys.argv[1], sys.argv[2], epochs=100)

p = {'optimizer': ['Adam'], #not used yet
     'losses': ['binary_crossentropy', dice_coef_loss],
     'epochs': [10, 20],
     'lr':[ 5e-5, 4e-5],
     'lr_decay': [0.0, 0.1, 0.25, 0.5],
     'model':["unet3D"],
     'stage2':[False]}



scan_object = talos.Scan(gen_3D, Null, model=HyperparameterTune, params=p,  experiment_name='unet3D', print_params=True)