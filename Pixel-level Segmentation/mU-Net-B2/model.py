import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras




def unet(pretrained_weights = None,input_size=(256,256,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(inputs)
    conv1_1 = BatchNormalization()(conv1)
    conv1_2 = Activation('relu')(conv1_1)
    conv2 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(conv1_2)
    conv2_1 = BatchNormalization()(conv2)
    conv2_2 = Activation('relu')(conv2_1)
    conv3 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(conv2_2)
    conv3_1 = BatchNormalization()(conv3)
    conv3_2 = Activation('relu')(conv3_1)
    conv4 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(inputs)
    conv4_1 = BatchNormalization()(conv4)
    conv4_2 = Activation('relu')(conv4_1)
    merge1 = concatenate([conv1_2, conv2_2, conv3_2, conv4_2], axis=3)
    down1 = MaxPooling2D(pool_size=(2, 2))(merge1)

    conv5 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(down1)
    conv5_1 = BatchNormalization()(conv5)
    conv5_2 = Activation('relu')(conv5_1)
    conv6 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv5_2)
    conv6_1 = BatchNormalization()(conv6)
    conv6_2 = Activation('relu')(conv6_1)
    conv7 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv6_2)
    conv7_1 = BatchNormalization()(conv7)
    conv7_2 = Activation('relu')(conv7_1)
    conv8 = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(down1)
    conv8_1 = BatchNormalization()(conv8)
    conv8_2 = Activation('relu')(conv8_1)
    merge2 = concatenate([conv5_2, conv6_2, conv7_2, conv8_2], axis=3)
    down2 = MaxPooling2D(pool_size=(2, 2))(merge2)

    conv9 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(down2)
    conv9_1 = BatchNormalization()(conv9)
    conv9_2 = Activation('relu')(conv9_1)
    conv10 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv9_2)
    conv10_1 = BatchNormalization()(conv10)
    conv10_2 = Activation('relu')(conv10_1)
    conv11 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv10_2)
    conv11_1 = BatchNormalization()(conv11)
    conv11_2 = Activation('relu')(conv11_1)
    conv12 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(down2)
    conv12_1 = BatchNormalization()(conv12)
    conv12_2 = Activation('relu')(conv12_1)
    merge3 = concatenate([conv9_2, conv10_2, conv11_2, conv12_2], axis=3)
    down3 = MaxPooling2D(pool_size=(2, 2))(merge3)

    conv13 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(down3)
    conv13_1 = BatchNormalization()(conv13)
    conv13_2 = Activation('relu')(conv13_1)
    conv14 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv13_2)
    conv14_1 = BatchNormalization()(conv14)
    conv14_2 = Activation('relu')(conv14_1)
    conv15 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv14_2)
    conv15_1 = BatchNormalization()(conv15)
    conv15_2 = Activation('relu')(conv15_1)
    conv16 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(down3)
    conv16_1 = BatchNormalization()(conv16)
    conv16_2 = Activation('relu')(conv16_1)
    merge4 = concatenate([conv13_2, conv14_2, conv15_2, conv16_2], axis=3)
    down4 = MaxPooling2D(pool_size=(2, 2))(merge4)

    conv17 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(down4)
    conv17_1 = BatchNormalization()(conv17)
    conv17_2 = Activation('relu')(conv17_1)
    conv18 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(conv17_2)
    conv18_1 = BatchNormalization()(conv18)
    conv18_2 = Activation('relu')(conv18_1)
    conv19 = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal')(conv18_2)
    conv19_1 = BatchNormalization()(conv19)
    conv19_2 = Activation('relu')(conv19_1)
    conv20 = Conv2D(256, (1, 1), padding='same', kernel_initializer='he_normal')(down4)
    conv20_1 = BatchNormalization()(conv20)
    conv20_2 = Activation('relu')(conv20_1)
    merge5 = concatenate([conv17_2, conv18_2, conv19_2, conv20_2], axis=3)


    up1 = Conv2DTranspose(1024, (2, 2), strides=2, padding='same', kernel_initializer='he_normal')(merge5)
    merge6 = concatenate([merge4, up1], axis=3)
    conv21 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(merge6)
    conv21_1 = BatchNormalization()(conv21)
    conv21_2 = Activation('relu')(conv21_1)
    conv22 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv21_2)
    conv22_1 = BatchNormalization()(conv22)
    conv22_2 = Activation('relu')(conv22_1)
    conv23 = Conv2D(128, (3, 3), padding='same', kernel_initializer='he_normal')(conv22_2)
    conv23_1 = BatchNormalization()(conv23)
    conv23_2 = Activation('relu')(conv23_1)
    conv24 = Conv2D(128, (1, 1), padding='same', kernel_initializer='he_normal')(merge6)
    conv24_1 = BatchNormalization()(conv24)
    conv24_2 = Activation('relu')(conv24_1)
    merge7 = concatenate([conv21_2, conv22_2, conv23_2, conv24_2], axis=3)

    up2 = Conv2DTranspose(512, (2, 2), strides=2, padding='same', kernel_initializer='he_normal')(merge7)
    merge8 = concatenate([merge3, up2], axis=3)
    conv25 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(merge8)
    conv25_1 = BatchNormalization()(conv25)
    conv25_2 = Activation('relu')(conv25_1)
    conv26 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv25_2)
    conv26_1 = BatchNormalization()(conv26)
    conv26_2 = Activation('relu')(conv26_1)
    conv27 = Conv2D(64, (3, 3), padding='same', kernel_initializer='he_normal')(conv26_2)
    conv27_1 = BatchNormalization()(conv27)
    conv27_2 = Activation('relu')(conv27_1)
    conv28 = Conv2D(64, (1, 1), padding='same', kernel_initializer='he_normal')(merge8)
    conv28_1 = BatchNormalization()(conv28)
    conv28_2 = Activation('relu')(conv28_1)
    merge9 = concatenate([conv25_2, conv26_2, conv27_2, conv28_2], axis=3)

    up3 = Conv2DTranspose(256, (2, 2), strides=2, padding='same', kernel_initializer='he_normal')(merge9)
    merge10 = concatenate([merge2, up3], axis=3)
    conv29 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(merge10)
    conv29_1 = BatchNormalization()(conv29)
    conv29_2 = Activation('relu')(conv29_1)
    conv30 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv29_2)
    conv30_1 = BatchNormalization()(conv30)
    conv30_2 = Activation('relu')(conv30_1)
    conv31 = Conv2D(32, (3, 3), padding='same', kernel_initializer='he_normal')(conv30_2)
    conv31_1 = BatchNormalization()(conv31)
    conv31_2 = Activation('relu')(conv31_1)
    conv32 = Conv2D(32, (1, 1), padding='same', kernel_initializer='he_normal')(merge10)
    conv32_1 = BatchNormalization()(conv32)
    conv32_2 = Activation('relu')(conv32_1)
    merge11 = concatenate([conv29_2, conv30_2, conv31_2, conv32_2], axis=3)

    up4 = Conv2DTranspose(128, (2, 2), strides=2, padding='same', kernel_initializer='he_normal')(merge11)
    merge12 = concatenate([merge1, up4], axis=3)
    conv33 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(merge12)
    conv33_1 = BatchNormalization()(conv33)
    conv33_2 = Activation('relu')(conv33_1)
    conv34 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(conv33_2)
    conv34_1 = BatchNormalization()(conv34)
    conv34_2 = Activation('relu')(conv34_1)
    conv35 = Conv2D(16, (3, 3), padding='same', kernel_initializer='he_normal')(conv34_2)
    conv35_1 = BatchNormalization()(conv35)
    conv35_2 = Activation('relu')(conv35_1)
    conv36 = Conv2D(16, (1, 1), padding='same', kernel_initializer='he_normal')(merge12)
    conv36_1 = BatchNormalization()(conv36)
    conv36_2 = Activation('relu')(conv36_1)
    merge13 = concatenate([conv33_2, conv34_2, conv35_2, conv36_2], axis=3)

    conv37 = Conv2D(1, 1, activation='sigmoid')(merge13)

    model = Model(input=inputs, output=conv37)

    # sgd = SGD(lr=1e-4 , momentum=0.9,decay=1e-6)

    # model.compile(optimizer = sgd, loss = 'dice', metrics = ['accuracy'])#SGD优化器
    # model.compile(optimizer=Adam(lr=0.0003), loss='dice', metrics=['accuracy'])#dice
    model.compile(optimizer = Adam(lr = 0.00015), loss = 'binary_crossentropy', metrics = ['accuracy'])  # 交叉熵
    # model.compile(optimizer=Adam(lr=0.00005), loss='dice', metrics=['accuracy'])
    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model



