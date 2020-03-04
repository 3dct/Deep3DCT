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
    img1 = sitk.ReadImage('E:\PWeinberger\Data\Probe3_1000.mhd')
    img2 = sitk.ReadImage('E:\PWeinberger\Data\Probe3_1000_bin_Inverted.mhd')
    normalizeFilter = sitk.NormalizeImageFilter()
    
    size_x = img1.GetWidth()
    size_y = img1.GetHeight()
    size_z = img1.GetDepth()

    Data_X = [] 
    Data_Y = []

    normalizedImage = normalizeFilter.Execute(img1)

    MirrorPaddingFilter = sitk.MirrorPadImageFilter()

    MirrorPaddingFilter.SetPadLowerBound([5,5,5])
    MirrorPaddingFilter.SetPadUpperBound([5,5,5])
    PaddedImage = MirrorPaddingFilter.Execute(normalizedImage)

    counter = 0

    for x in range(0, size_x-122, 122):
            for y in range(0, size_y-122, 122):
                for z in range(0, size_z-122, 122):
                    image = sitk.RegionOfInterest(PaddedImage,(132,132,132),(x,y,z))
                    mask = sitk.RegionOfInterest(img2,(122,122,122),(x,y,z))

                    #sitk.WriteImage(image,'Train/Image'+str(counter)+".mhd")
                    #sitk.WriteImage(mask,'Train/Mask'+str(counter)+".mhd")

                    image = sitk.GetArrayFromImage(image)
                    MaskSeg = sitk.GetArrayFromImage(mask)

                    ImageTrain = np.reshape(image,(1,132,132,132,1))
                    MaskTrain = np.reshape(MaskSeg,(1,122,122,122,1))

                    Data_X.append(ImageTrain)
                    Data_Y.append(MaskTrain)

                    counter = counter +1

        
    
    Data_X = np.reshape(Data_X,(counter,132,132,132,1))
    Data_Y = np.reshape(Data_Y,(counter,122,122,122,1))

    return Data_X, Data_Y