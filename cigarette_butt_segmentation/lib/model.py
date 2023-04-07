from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Conv2DTranspose, Concatenate, Reshape
from tensorflow.keras.models import Model
from keras import backend as K


def block_down(c, filter, kernel):
    """
    The block down of the UNet structure
    Parameters
    ----------
    c - Tensorflow tensor
    filter - number of filters in the output tensor
    kernel - kernel of convolution

    Returns
    -------
    c - Tensorflow tensor after two convolutions
    p - MaxPooled tensor c
    """
    c = Conv2D(filter, kernel, padding='same', activation='relu')(c)
    c = Conv2D(filter, kernel, padding='same', activation='relu')(c)
    p = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(c)
    return p, c


def block_up(c, filter, kernel):
    """

    Parameters
    ----------
     c - Tensorflow tensor
    filter - number of filters in the output tensor
    kernel - kernel of convolution

    Returns
    -------
    c - Tensorflow tensor after two convolutions
    p - UpSampled by transpose convolution tensor c
    -------

    """
    c = Conv2D(filter, kernel, padding='same', activation='relu')(c)
    c = Conv2D(filter, kernel, padding='same', activation='relu')(c)
    c = Conv2DTranspose(filter // 2, kernel_size=(2, 2), strides=(2, 2))(c)
    return c


def final_block(c, filter, kernel, activation):
    """
    Final convolution of UNet with the sigmoid activation
    Parameters
    ----------
     c - Tensorflow tensor
    filter - number of filters in the output tensor
    kernel - kernel of convolution

    Returns
    -------
    c - Tensorflow tensor after two convolutions
    -------

    """
    c = Conv2D(filter, kernel, padding='same', activation='relu')(c)
    c = Conv2D(filter, kernel, padding='same', activation='relu')(c)
    c = Conv2D(1, kernel_size=(2, 2), padding='same', activation=activation)(c)
    return c

def dice(y_true,y_preds,thr=0.5):
    """
    Dice metric for evaluation of training. Dice is double intersection over union, if prediction totally correct it is 1, and 0 - for totally wrong
    Parameters
    ----------
    y_true - Theano Tensor
    y_pred - Theano Tensor

    Returns
    -------

    """
    y_preds = K.cast_to_floatx(K.greater(y_preds,thr))
    intersection = K.sum(y_true*y_preds, axis=[1,2,3])
    union = K.sum(y_preds, axis=[1,2,3]) + K.sum(y_true, axis=[1,2,3]) + K.epsilon()
    dices = 2 * intersection / union
    return K.mean(dices)


def get_model(loss_fn, activation_last_layer='sigmoid', metrics=[dice], fst_filter=32):
    """

    Parameters
    ----------
    loss_fn - loss function of model
    activation_last_layer - activation of the last layer
    metrics - metrics for evaluation of model
    fst_filter - the scale of UNet. The original UNet structure starts from 64 filters, if fst_filter is equal 32 - it decreases twice the UNet parameters

    Returns
    -------
    Model of the Tensorflow framework with UNet structure
    """
    input_layer = Input(shape=(512, 512, 3))

    out, p1 = block_down(input_layer, fst_filter, (3, 3))  # 512 -> fst_filter * 4
    out, p2 = block_down(out, fst_filter * 2, (3, 3))  # fst_filter * 4 -> fst_filer * 2
    out, p3 = block_down(out, fst_filter * 4, (3, 3))  # fst_filer * 2 -> fst_filter
    out, p4 = block_down(out, fst_filter * 8, (3, 3))  # fst_filter -> 32

    out = block_up(out, fst_filter * 16, (3, 3))  # fst_filter -> fst_filter

    out = Concatenate()([p4, out])
    out = block_up(out, fst_filter * 8, (3, 3))  # fst_filter -> fst_filer * 2

    out = Concatenate()([p3, out])
    out = block_up(out, fst_filter * 4, (3, 3))  # fst_filer * 2 -> fst_filter * 4

    out = Concatenate()([p2, out])
    out = block_up(out, fst_filter * 2, (3, 3))  # fst_filter * 4 -> 512

    out = Concatenate()([p1, out])
    out = final_block(out, fst_filter, (3, 3), activation=activation_last_layer)

    model = Model(input_layer, out)

    model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)

    return model
