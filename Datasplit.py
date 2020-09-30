import DataLoad 
import numpy as np
import SimpleITK as sitk
from sklearn.utils import shuffle


InputData = "Volume_340x270x270\pore_1.mhd"
Label = "Volume_340x270x270\label1.mhd"

numberOfChunks = 8
Outputname = "Training"

def transformSave(ImgArray, index=0, shape=(128,128,128), name='Result_',dir="Out\\"):
    img = np.reshape(ImgArray,shape)
    outImg = sitk.GetImageFromArray(img)
    sitk.WriteImage(outImg,dir + "\\" + name + str(index) + ".mhd")


X,Y = DataLoad.load3D_XY(InputData, Label,False,0)

X,Y = shuffle(X,Y)


chunkSize = round(X.shape[0] / numberOfChunks)

dirTrain = ''
dirLabel = ''

for i in range(X.shape[0]):

    if i%chunkSize == 0:
        import pathlib

        dirTrain = Outputname + "\\Train\\Train_%02d" % (i/chunkSize)
        pathlib.Path(dirTrain).mkdir(parents=True, exist_ok=True) 

        dirLabel = Outputname + "\\Label\\Label_%02d" % (i/chunkSize)
        pathlib.Path(dirLabel).mkdir(parents=True, exist_ok=True) 

    transformSave(X[i],i,(132,132,132),'Train_',dirTrain)
    transformSave(Y[i],i,(122,122,122),'Label_',dirLabel)


