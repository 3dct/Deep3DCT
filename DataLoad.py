import SimpleITK as sitk
import numpy as np
import Show 

def load3D():
    img1 = sitk.ReadImage('Z:\PWeinberger\Probe3_1000.mhd')
    img2 = sitk.ReadImage('Z:\PWeinberger\Probe3_1000_bin.mhd')
    normalizeFilter = sitk.NormalizeImageFilter()
    for i in range(0,25,1):
        for x in range(0, 870, 128):
            for y in range(0, 870, 128):
                for z in range(0, 870, 128):
                    image = sitk.RegionOfInterest(img1,(128,128,128),(x,y,z))
                    mask = sitk.RegionOfInterest(img2,(128,128,128),(x,y,z))

                    sitk.WriteImage(mask,"Mask_seg.mhd")

                    normalizedImage = sitk.GetArrayFromImage(normalizeFilter.Execute(image))
                    MaskSeg = sitk.GetArrayFromImage(mask)

                    ImageTrain = np.reshape(normalizedImage,(1,128,128,128,1))
                    MaskTrain = np.reshape(MaskSeg,(1,128,128,128,1))

                    Show.transformSave(MaskTrain)

                    yield (ImageTrain,MaskTrain)


def load3D_XY():
    img1 = sitk.ReadImage('Z:\PWeinberger\Probe3_1000.mhd')
    img2 = sitk.ReadImage('Z:\PWeinberger\Probe3_1000_bin.mhd')
    normalizeFilter = sitk.NormalizeImageFilter()

    Data_X = [] 
    Data_Y = []

    counter = 0

    for i in range(0,25,1):
        for x in range(0, 870, 128):
            for y in range(0, 870, 128):
                for z in range(0, 870, 128):
                    image = sitk.RegionOfInterest(img1,(128,128,128),(x,y,z))
                    mask = sitk.RegionOfInterest(img2,(128,128,128),(x,y,z))

                    sitk.WriteImage(mask,"Mask_seg.mhd")

                    normalizedImage = sitk.GetArrayFromImage(normalizeFilter.Execute(image))
                    MaskSeg = sitk.GetArrayFromImage(mask)

                    ImageTrain = np.reshape(normalizedImage,(1,128,128,128,1))
                    MaskTrain = np.reshape(MaskSeg,(1,128,128,128,1))

                    Data_X.append(ImageTrain)
                    Data_Y.append(MaskTrain)

                    counter = counter +1
    
    Data_X = np.reshape(Data_X,(counter,128,128,128,1))
    Data_Y = np.reshape(Data_Y,(counter,128,128,128,1))

    return Data_X, Data_Y