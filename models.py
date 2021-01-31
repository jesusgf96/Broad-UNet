import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Add, Dropout, concatenate, BatchNormalization, Activation, Multiply, Lambda, Reshape
from tensorflow.keras.layers import Conv3D, MaxPool3D, UpSampling3D, GlobalAveragePooling3D
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D
    
    
def UNet_original(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):    
    
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (latitude, longitude, features)) 
    conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = Conv2D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = Conv2D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = Conv2D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv4)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = Conv2D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    drop5 = Dropout(dropout)(conv5)
    
    #--- Expanding part / decoder ---#
    up6 = Conv2D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = -1)
    conv6 = Conv2D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = Conv2D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv6)

    up7 = Conv2D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = -1)
    conv7 = Conv2D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = Conv2D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv7)

    up8 = Conv2D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = -1)
    conv8 = Conv2D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = Conv2D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv8)

    up9 = Conv2D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = -1)
    conv9 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = Conv2D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv9 = Conv2D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv2D(features_output, 1, activation = 'sigmoid', padding = 'same')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)

    
    
    
def UNet_AsymmetricInceptionRes3DDR(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):    

    def res_inception_block(x, f, k):
        shortcut=x
        x1 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x2 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x3 = Conv3D(f, (1,1,5), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x3 = Conv3D(f, (1,5,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x3 = Conv3D(f, (5,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x4 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x4 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x4)
        x4 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x4)
        x5 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x5 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x5)
        x5 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x5)
        x = concatenate([x1, x2, x3, x4, x5], axis = -1)
        x = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = res_inception_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = res_inception_block(conv2, 2*filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = res_inception_block(conv3, 4*filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = res_inception_block(conv4, 8*filters, 3)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = res_inception_block(conv5, 16*filters, 3)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    drop5 = Dropout(dropout)(compressLags)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = res_inception_block(conv6, 8*filters, 3)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = res_inception_block(conv7, 4*filters, 3)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = res_inception_block(conv8, 2*filters, 3)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = res_inception_block(conv9, filters, 3)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'sigmoid', padding = 'same')(conv9) #Reduce last dimension    

    return Model(inputs = inputs, outputs = conv10)



def broad_UNet(lags, latitude, longitude, features, features_output, filters, dropout, kernel_init=tf.keras.initializers.GlorotUniform(seed=50)):    

    def image_level_feature_pooling(x, f):
        up_size = x.get_shape().as_list()[2:4]
        x = GlobalAveragePooling3D()(x)
        x = Reshape((1, 1, 1, f))(x)
        x = UpSampling3D(size = (1,up_size[0],up_size[1]))(x)
        return x


    def ASPP_block(x, f):
        x1 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1, 3, 3), activation = 'relu', padding = 'same', dilation_rate=6,  kernel_initializer = kernel_init)(x)
        x3 = Conv3D(f, (1, 3, 3), activation = 'relu', padding = 'same', dilation_rate=12,  kernel_initializer = kernel_init)(x)
        x4 = Conv3D(f, (1, 3, 3), activation = 'relu', padding = 'same', dilation_rate=18,  kernel_initializer = kernel_init)(x)
        x5 = image_level_feature_pooling(x, f)
        x = concatenate([x1, x2, x3, x4, x5], axis = -1)
        x = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        return x


    def convolutional_block(x, f, k):
        shortcut=x
        x1 = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x2 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x2 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x2)
        x3 = Conv3D(f, (1,1,5), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x3 = Conv3D(f, (1,5,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x3 = Conv3D(f, (5,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x3)
        x4 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x4 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x4)
        x4 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x4)
        x5 = Conv3D(f, (1,1,3), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x5 = Conv3D(f, (1,3,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x5)
        x5 = Conv3D(f, (3,1,1), activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x5)
        x = concatenate([x1, x2, x3, x4, x5], axis = -1)
        x = Conv3D(f, 1, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(x)
        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x
    
    #--- Contracting part / encoder ---#
    inputs = Input(shape = (lags, latitude, longitude, features)) 
    conv1 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(inputs)
    conv1 = convolutional_block(conv1, filters, 3)
    pool1 = MaxPool3D(pool_size=(1, 2, 2))(conv1)
    
    conv2 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool1)
    conv2 = convolutional_block(conv2, 2*filters, 3)
    pool2 = MaxPool3D(pool_size=(1, 2, 2))(conv2)
    
    conv3 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool2)
    conv3 = convolutional_block(conv3, 4*filters, 3)
    pool3 = MaxPool3D(pool_size=(1, 2, 2))(conv3)
    
    conv4 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool3)
    conv4 = convolutional_block(conv4, 8*filters, 3)
    drop4 = Dropout(dropout)(conv4)
    
    #--- Bottleneck part ---#
    pool4 = MaxPool3D(pool_size=(1, 2, 2))(drop4)
    conv5 = Conv3D(16*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(pool4)
    conv5 = convolutional_block(conv5, 16*filters, 3)
    compressLags = Conv3D(16*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv5)
    aspp = ASPP_block(compressLags, 16*filters)
    drop5 = Dropout(dropout)(aspp)
    
    #--- Expanding part / decoder ---#
    up6 = Conv3D(8*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(drop5))
    compressLags = Conv3D(8*filters, (lags,1,1),activation = 'relu', padding = 'valid')(drop4)
    merge6 = concatenate([compressLags,up6], axis = -1)
    conv6 = Conv3D(8*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge6)
    conv6 = convolutional_block(conv6, 8*filters, 3)

    up7 = Conv3D(4*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv6))
    compressLags = Conv3D(4*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv3)
    merge7 = concatenate([compressLags,up7], axis = -1)
    conv7 = Conv3D(4*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge7)
    conv7 = convolutional_block(conv7, 4*filters, 3)

    up8 = Conv3D(2*filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv7))
    compressLags = Conv3D(2*filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv2)
    merge8 = concatenate([compressLags,up8], axis = -1)
    conv8 = Conv3D(2*filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge8)
    conv8 = convolutional_block(conv8, 2*filters, 3)

    up9 = Conv3D(filters, 2, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(UpSampling3D(size = (1,2,2))(conv8))
    compressLags = Conv3D(filters, (lags,1,1),activation = 'relu', padding = 'valid')(conv1)
    merge9 = concatenate([compressLags,up9], axis = -1)
    conv9 = Conv3D(filters, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(merge9)
    conv9 = convolutional_block(conv9, filters, 3)
    conv9 = Conv3D(2*features_output, 3, activation = 'relu', padding = 'same', kernel_initializer = kernel_init)(conv9)
    
    conv10 = Conv3D(features_output, 1, activation = 'sigmoid', padding = 'same')(conv9) #Reduce last dimension     

    return Model(inputs = inputs, outputs = conv10)

