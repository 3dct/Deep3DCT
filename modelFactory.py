from model import unet, unet3D
from VnetModel import vnet3D
from keras.models import *
from keras.optimizers import *
from keras.utils import multi_gpu_model
from metrics import dice_coef_loss



def getModel(modelName = 'unet3D'):

    model = []

    switcher = {
        "unet": unet(),
        "unet3D": unet3D(),
        "vnet3D": vnet3D(),
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

    execute_model.compile(optimizer = Adam(lr = 3e-3, decay=0.5), loss = dice_coef_loss, metrics = ['accuracy'])
    execute_model.summary()

    return execute_model

