import DataLoad 
import numpy as np
import SimpleITK as sitk
from sklearn.utils import shuffle

def transformSave(ImgArray, index=0, shape=(128,128,128), name='Result_',dir="Out\\"):
    img = np.reshape(ImgArray,shape)
    outImg = sitk.GetImageFromArray(img)
    sitk.WriteImage(outImg,dir + "\\" + name + str(index) + ".mhd")


X,Y = DataLoad.load3D_XY("E:\\Julia\\CR-sGF24-1-2um-396N-new-bhc8-16bit-1587x1105x1800.mhd", "E:\\Julia\\396N-DefectVis-bin.mhd",False,100)

X,Y = shuffle(X,Y)

numberOfChunks = 20
chunkSize = round(X.shape[0] / numberOfChunks)

dirTrain = ''
dirLabel = ''

for i in range(X.shape[0]):

    if i%chunkSize == 0:
        import pathlib

        dirTrain = "396N_without2\\Train\\Train_%02d" % (i/chunkSize)
        pathlib.Path(dirTrain).mkdir(parents=True, exist_ok=True) 

        dirLabel = "396N_without2\\Label\\Label_%02d" % (i/chunkSize)
        pathlib.Path(dirLabel).mkdir(parents=True, exist_ok=True) 

    transformSave(X[i],i,(132,132,132),'Train_',dirTrain)
    transformSave(Y[i],i,(122,122,122),'Label_',dirLabel)


