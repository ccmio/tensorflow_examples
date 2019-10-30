import numpy as np
from models_examples.cnn import CNN
import tensorflow as tf


class MNISTLoader:
    # 全体数据预处理
    def __init__(self):
        # 载入数据集对象mnist
        mnist = tf.keras.datasets.mnist
        # 数据集分装至train/test，data/label
        # train_data为60000个28*28的数组表示的图片，test_data为10000个28*28
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()

        # MNIST中的图像默认为uint8（0-255的数字）。所有像素点/255使其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)      # [60000, 28, 28, 1]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)        # [10000, 28, 28, 1]
        self.train_label = self.train_label.astype(np.int32)    # [60000]
        self.test_label = self.test_label.astype(np.int32)      # [10000]
        # 提醒自己，不要用len，用shape[0]
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    # 取训练batch（data与label）
    def get_batch(self, batch_size):
        # index为从0到60000之间随机的batch_size个数，并不是连续的
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        # 按照index存储的batch_size个数，从train里面取出训练单次训练batch
        return self.train_data[index, :], self.train_label[index]


# 模型构建
# class MLP(tf.keras.Model):
#     def __init__(self):
#         super().__init__()
#         self.flatten = tf.keras.layers.Flatten()    # Flatten层将除第一维（batch_size）以外的维度展平
#         self.dense1 = tf.keras.layers.Dense(units=10, activation=tf.nn.relu)
#         self.dense2 = tf.keras.layers.Dense(units=100)
#         self.dense3 = tf.keras.layers.Dense(units=10)
#
#     def call(self, inputs):         # [batch_size, 28, 28, 1]
#         x = self.flatten(inputs)    # [batch_size, 784]
#         x = self.dense1(x)          # [batch_size, 10]
#         x = self.dense2(x)          # [batch_size, 100]
#         x = self.dense3(x)          # [batch_size, 10]
#         output = tf.nn.softmax(x)
#         return output


def main():
    # 超参数
    num_epochs = 5
    batch_size = 500  # 每批次训练的batch大小
    learning_rate = 0.001  # 学习率

    model = CNN()

    # 获得数据集，并预处理完毕，分发至train/test
    data_loader = MNISTLoader()
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    num_batches = int(data_loader.num_train_data // batch_size)
    # 进行训练
    for i in range(num_epochs):
        for batch_index in range(num_batches):
            # 单次训练data与label取得
            X, y = data_loader.get_batch(batch_size)
            # 梯度下降
            with tf.GradientTape() as tape:
                y_pred = model(X)
                # sparse_categorical_crossentropy 交叉熵损失函数
                loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
                loss = tf.reduce_mean(loss)
                print("batch %d: loss %f" % (batch_index, loss.numpy()))
            # 算出梯度
            grads = tape.gradient(loss, model.variables)
            # 使用优化器进行梯度下降
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

        # 进行评估
        # SparseCategoricalAccuracy()计算准确率
        sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

        num_batches = int(data_loader.num_test_data // batch_size)
        for batch_index in range(num_batches):
            start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
            y_pred = model.predict(data_loader.test_data[start_index: end_index])
            sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
        print("test accuracy: %f" % sparse_categorical_accuracy.result())


if __name__ == '__main__':
    main()
