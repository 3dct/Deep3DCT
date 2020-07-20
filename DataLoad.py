import SimpleITK as sitk
import numpy as np
import Show 

def load3D(image, mask, numberOfImages=1000, epochs=10, batch_size=1):
    img1 = sitk.ReadImage(image)
    img2 = sitk.ReadImage(mask)
    normalizeFilter = sitk.NormalizeImageFilter()

    size_x = img1.GetWidth()
    size_y = img1.GetHeight()
    size_z = img1.GetDepth()

    MirrorPaddingFilter = sitk.MirrorPadImageFilter()

    MirrorPaddingFilter.SetPadLowerBound([5,5,5])
    MirrorPaddingFilter.SetPadUpperBound([5,5,5])
    PaddedImage = MirrorPaddingFilter.Execute(normalizeFilter.Execute(img1))


    for i in range(0,epochs,1):

        counter = 0

        ImageTrain = None
        MaskTrain = None

        for x in range(0, size_x-121, 122):
            for y in range(0, size_y-121, 122):
                for z in range(0, size_z-121, 122):
                    image = sitk.RegionOfInterest(PaddedImage,(132,132,132),(x,y,z))
                    mask = sitk.RegionOfInterest(img2,(122,122,122),(x,y,z))

                    #sitk.WriteImage(image,'Train/Image'+str(counter)+".mhd")
                    #sitk.WriteImage(mask,'Train/Mask'+str(counter)+".mhd")

                    image = sitk.GetArrayFromImage(image)
                    MaskSeg = sitk.GetArrayFromImage(mask)

                    voxelNotzero = np.count_nonzero(MaskSeg)


                    if (voxelNotzero>20 or counter%9==0):
                        ImageTrain = np.reshape(image,(1,132,132,132,1))
                        MaskTrain = np.reshape(MaskSeg,(1,122,122,122,1))

                        counter = counter +1

                    
                        if ((counter<=numberOfImages)) :
                            yield (ImageTrain,MaskTrain)


def load3D_XY(image, mask):
    img1 = sitk.ReadImage(image)
    img2 = sitk.ReadImage(mask)
    normalizeFilter = sitk.NormalizeImageFilter()

    size_x = img1.GetWidth()
    size_y = img1.GetHeight()
    size_z = img1.GetDepth()

    size_x =  size_x if size_x <= 400 else 400
    size_y =  size_y if size_y <= 400 else 400
    size_z =  size_z if size_z <= 600 else 600

    Data_X = [] 
    Data_Y = []



    MirrorPaddingFilter = sitk.MirrorPadImageFilter()

    MirrorPaddingFilter.SetPadLowerBound([5,5,5])
    MirrorPaddingFilter.SetPadUpperBound([5,5,5])
    PaddedImage = MirrorPaddingFilter.Execute(normalizeFilter.Execute(img1))

    img1=0


    counter = 0

    for x in range(0, size_x-121, 122):
            for y in range(0, size_y-121, 122):
                for z in range(0, size_z-121, 122):
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