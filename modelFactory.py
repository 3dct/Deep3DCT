from model import unet, unet3D
from VnetModel import vnet3D
from keras.models import *
from keras.optimizers import *
from keras.utils import multi_gpu_model



def getModel(modelName = 'unet3D'):

    model = []

    switcher = {
        "unet": unet(),
        "unet3D": unet3D(),
        "vnet3D": vnet3D(),
    }

    selected_model = switcher.get(modelName)

    try:
        execute_model = multi_gpu_model(selected_model, cpu_relocation=True)
        print("Training using multiple GPUs..")
    except ValueError:
        execute_model = selected_model
        print("Training using single GPU or CPU..")


    execute_model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    execute_model.summary()

    return execute_model

