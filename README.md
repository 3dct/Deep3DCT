# Implementation of deep learning framework -- Unet, using Keras

The architecture was inspired by [U-Net: Convolutional Networks for Biomedical Image Segmentation](http://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/).

---

## Overview


### Model


This deep neural network is implemented with Keras functional API, which makes it extremely easy to experiment with different interesting architectures.

The input has the size of 132x132x132. (Overlapping of input due smaller output size)
Output from the network is a 122x122x122 which represents mask that should be learned. Sigmoid activation function
makes sure that mask pixels are in \[0, 1\] range.

### Prepare Data for training

Execute Datasplitt.py to splitt a data set to the needed samples for training.
To do this a mirror padding is added and the data is splitted. As input a mhd Volume is used.

### Training

The model is trained for 20 epochs.

After 20 epochs, calculated accuracy is about 0.98.

Loss function for the training is a focal Tversky loss used.


---

### Dependencies


* Tensorflow
* Keras >= 2.3
* keras2onnx
* sklearn
* simpleITK
* talos (Hyperparameter optimization)

### Run main.py




