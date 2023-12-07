import keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    """
    Calculate the Dice coefficient for binary segmentation.
    :param y_true: Ground truth binary segmentation mask.
    :param y_pred: Predicted binary segmentation mask.
    :param smooth: Smoothing factor to prevent division by zero.
    :return: Dice coefficient value
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)


def IoU(y_true, y_pred):
    """
    Calculate the IoU coefficient
    :param y_true: Ground truth binary segmentation mask.
    :param y_pred: Predicted binary segmentation mask.
    :return: Dice coefficient value
    """
    y_pred_f = K.flatten(y_pred)
    y_true_f = K.flatten(y_true)
    intersection = K.sum(y_true_f * y_pred)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return union / intersection
