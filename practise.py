import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def AlexNet_inference(in_shape):
    model = keras.Sequential(name='AlexNet')

    # model.add(layers.Conv2D(96,(11,11),strides=(4,4),input_shape=(in_shape[1],in_shape[2],in_shape[3]),
    # padding='same',activation='relu',kernel_initializer='uniform'))

    model.add(layers.Conv2D(96, (11, 11), strides=(2, 2), input_shape=(in_shape[1], in_shape[2], in_shape[3]),
                            padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(
        layers.Conv2D(256, (5, 5), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(
        layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(
        layers.Conv2D(384, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(
        layers.Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='uniform'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(2048, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer=keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',  # 不能直接用函数，否则在与测试加载模型不成功！
                  metrics=['accuracy'])
    model.summary()

    return model


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


MODEL_DIR = "models/"
DataSetPath = "dataSets/fashion/"
# DataSetPath = "dataSets/mnist/"
x_train, y_train = mnist_reader.load_mnist(DataSetPath, 'train')
x_test, y_test = mnist_reader.load_mnist(DataSetPath, 't10k')

x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

print(x_train.shape[1], x_train.shape[2], x_train.shape[3])

x_shape = x_train.shape


def AlexNet_train():
    # 加载与训练权重
    # AlexNet_model = keras.models.load_model(PRE_MODEL_DIR)

    AlexNet_model = AlexNet.AlexNet_inference(x_shape)
    totall_epochs = 0
    epochs = 10

    while (True):

        history = AlexNet_model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_split=0.1)

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.legend(['training', 'valivation'], loc='upper left')
        plt.show()

        res = AlexNet_model.evaluate(x_test, y_test)
        print(res)

        totall_epochs += epochs
        model_save_dir = MODEL_DIR + 'AlexNet_model_' + str(totall_epochs) + '.h5'
        AlexNet_model.save(model_save_dir)

        keyVal = input('please enter your command!(0:quite, 1>:continue!)')
        keyVal = int(keyVal)
        if 0 == keyVal:
            break
        elif 0 <= keyVal and 10 >= keyVal:
            epochs = keyVal
