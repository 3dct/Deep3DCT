from modelFactory import HyperparameterTune
from metrics import dice_coef_loss
from BoundaryLoss import surface_loss, gl_sl_wrapper

import DataLoad 
from sklearn.model_selection import train_test_split
import talos
import sys

import tensorflow as tf
from tensorflow import keras
from FocalTverskyLoss import FocalTverskyLoss
print(tf.version.VERSION)

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

X,Y = DataLoad.load3D_XY('E:\DATA\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man8bit+VS_Calibrated_1220x854x976.mhd',
    'E:\DATA\AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976\General_otsu_BIN_AI_Referenz_CFK_3_3um_Probe2_60kV_noinlMED-BHC0_man16bit+VS_Calibrated_1220x854x976.mhd')

#X,Y = DataLoad.load3D_XY(sys.argv[1], sys.argv[2])

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

#gen_3D = DataLoad.load3D(sys.argv[1], sys.argv[2], epochs=100)

p = {'optimizer': ['Adam', 'Nadam'], #not used yet
     #'losses': ['binary_crossentropy', dice_coef_loss],
     'losses': [FocalTverskyLoss],
     'epochs': [10],
     'lr':( 5e-6, 1e-4,20),
     'lr_decay': [0.0],
     'model':["unet3D"],
     'stage2':[True],
     'alpha':(0.0, 0.9,10),
     'gamma':[1]}



scan_object = talos.Scan(X, Y, model=HyperparameterTune, params=p,  experiment_name='unet3D', print_params=True)