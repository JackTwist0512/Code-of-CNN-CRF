import time
import math, json, os, sys
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import pandas as pd
import tensorflow as tf
# num_classes=2
# def mycrossentropy(y_true, y_pred, e=0.14):
#        return (1-e)*K.categorical_crossentropy(y_pred,y_true) + e*K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/num_classes)

"""
#1,数据路径
#2，模型保存路径名称,注意更改加载模型
#3，修改存储 history 路径
#4，存储fig 的标题以及路径
"""
dataset = 3
model_name = 3
times = 3

DATA_DIR = 'C:\\Users\\zjh\\Desktop\\smear2005\\dataset'+str(dataset)+'\\'

TRAIN_DIR = os.path.join(DATA_DIR, 'train_')

VALID_DIR = os.path.join(DATA_DIR, 'validation_')

SIZE = (224, 224)
BATCH_SIZE = 16




def save_history(History):
    acc = pd.Series(History.history['acc'], name='acc')
    loss = pd.Series(History.history['loss'], name='loss')
    val_acc = pd.Series(History.history['val_acc'], name='val_acc')
    val_loss = pd.Series(History.history['val_loss'], name='val_loss')
    com = pd.concat([acc, loss, val_acc, val_loss], axis=1)
    # 注意存储位置！！
    com.to_csv('C:\\Users\\zjh\\Desktop\\smear2005\\dataset'+str(dataset)+'\\model\\history_'+str(model_name)+'_'+str(times)+'.csv')


# 画出acc loss曲线
def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # 在此处更改 图标的标题！！
    plt.title("vgg16 model")
    plt.ylabel("acc-loss")
    plt.xlabel("epoch")
    plt.legend([" acc", "val acc", " loss", "val loss"], loc="upper right")
    # plt.show()
    # 注意修改存储名称！！！
    plt.savefig('C:\\Users\\zjh\\Desktop\\smear2005\\dataset'+str(dataset)+'\\model\\model_'+str(model_name)+'_'+str(times)+'.png')


if __name__ == "__main__":
    start = time.clock()
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples / BATCH_SIZE)
    num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator()
    #val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator()

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True,
                                      batch_size=BATCH_SIZE)
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True,
                                              batch_size=BATCH_SIZE)

    classes = list(iter(batches.class_indices))
    Inp = Input((224, 224, 3))
    if model_name == 1:
        base_model = keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name ==2:
        base_model = keras.applications.inception_v3.InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name == 3:
        base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    elif model_name ==4:
        base_model = keras.applications.xception.Xception(weights='imagenet', include_top=False, input_shape=(224, 224, 3))




    x = base_model(Inp)
    # x = base_model(Inp).layers[-1].output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(len(classes), activation="softmax")(x)
    model = Model(inputs=Inp, outputs=predictions)
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    # for c in batches.class_indices:
    #     classes[batches.class_indices[c]] = c
    # finetuned_model.classes = classes

    # early_stopping = EarlyStopping(patience=15)

    checkpointer = ModelCheckpoint('C:\\Users\\zjh\\Desktop\\smear2005\\dataset'+str(dataset)+'\\model\\model'+str(model_name)+'_time'+str(times)+'.h5',
                                   verbose=1, save_best_only=True)

    History = model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=100,
                                            callbacks=[ checkpointer], validation_data=val_batches,
                                            validation_steps=num_valid_steps)

    model.summary()
    # end1 = time.clock()
    # print("save model before", end1)
    # finetuned_model.save('D:\\U5510\\Random1\\VGG16_26\\0.75\\model\\vgg16_best_1.h5')
    save_history(History)
    plot_history(History)

    end2 = time.clock()
    print("final is in ", end2 - start)