from model import unet, unet3D
from VnetModel import vnet3D
from keras.models import *
from keras.optimizers import *
from keras.utils import multi_gpu_model
from metrics import dice_coef_loss, weighted_dice_coefficient_loss, f1_m,precision_m, recall_m, true_positives, predicted_positives
from keras import backend as K
from BoundaryLoss import surface_loss, gl_sl_wrapper
from FocalTverskyLoss import FocalTverskyLoss

import talos





def getModel(modelName = 'unet3D'):



    model = []

    switcher = {
        "unet": unet(),
        "unet3D": unet3D(),
        #"vnet3D": vnet3D(),
    }

    selected_model = switcher.get(modelName)

    #Not working bug in tensorflow
    # try:
    #     execute_model = multi_gpu_model(selected_model, cpu_relocation=True)
    #     print("Training using multiple GPUs..")
    # except ValueError:
    #     execute_model = selected_model
    #     print("Training using single GPU or CPU..")


    execute_model = selected_model

    execute_model.compile(optimizer = Nadam(lr = 8e-5), loss = FocalTverskyLoss, metrics = ['accuracy', f1_m,precision_m, recall_m, true_positives, predicted_positives])
    execute_model.summary()

    return execute_model


def HyperparameterTune(x_train, y_train, x_val, y_val, params):

    model = []

    switcher = {
        #"unet": unet(),
        "unet3D": unet3D(stage2=params["stage2"]),
        #"vnet3D": vnet3D(),
    }

    selected_model = switcher.get(params["model"])

    execute_model = selected_model

    if params["optimizer"] == "Adam":
        optimizer = Adam(lr = params["lr"], decay=params["lr_decay"])
    else:
        optimizer = Nadam(lr = params["lr"])

    execute_model.compile(optimizer = optimizer, loss = [lambda y_true,y_pred: FocalTverskyLoss(y_true,y_pred,params["alpha"],1-params["alpha"],params["gamma"])] , metrics = ['accuracy', f1_m])
    #execute_model.summary()
    from talos.utils import early_stopper

    out = execute_model.fit(x_train, y_train,batch_size=3,epochs=params["epochs"], validation_data=(x_val, y_val),
                     verbose=0, callbacks = [early_stopper(epochs=params['epochs'],  
                                                    monitor='val_accuracy')])

    return out, execute_model
    

