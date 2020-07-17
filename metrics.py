from functools import partial

from keras import backend as K

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



def dice_coefficient_loss(y_true, y_pred):
    return 1 -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(0, 1, 2,3), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f

def recall_m(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    true_positives = K.sum(y_true_f * y_pred_f)
    possible_positives = K.sum(y_true_f) + K.epsilon()
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred) 
    true_positives = K.sum(y_true_f * y_pred_f)
    predicted_positives = K.sum(y_pred_f) + K.epsilon()
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def true_positives(y_true, y_pred):
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred) 
    true_positives = K.sum(y_true_f * y_pred_f) + K.epsilon()
    return true_positives

def predicted_positives(y_true, y_pred):
    y_true_f = K.flatten(y_true) 
    y_pred_f = K.flatten(y_pred) 
    predicted_positives = K.sum(y_pred_f) + K.epsilon()
    return predicted_positives

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return -2*((precision*recall)/(precision+recall+K.epsilon()))

def dice1(y_true, y_pred, eps=1e-3):
    weights = 1./(K.sum(y_true, axis=[0,1,2,3])+eps)
    weights = weights/K.sum(weights)*0.3
    num = K.sum(weights*K.sum(y_true*y_pred, axis=[0,1,2,3]))
    den = K.sum(weights*K.sum(y_true+y_pred, axis=[0,1,2,3]))
    return 2.*(num+eps)/(den+eps)

def dice_loss(y_true, y_pred):
    return 1-dice1(y_true, y_pred)

def weights(y_true, y_pred, eps=1e-3):
    weights = 1./(K.sum(y_true, axis=[0,1,2,3])+eps)
    #weights = weights/K.sum(weights)
    return weights

dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss
