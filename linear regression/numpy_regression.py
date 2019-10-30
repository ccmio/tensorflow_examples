import numpy as np
import matplotlib.pyplot as plt

lr = 0.1
epoch = 100
a, b = 0, 0
x_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

x = (x_raw - x_raw.min())/(x_raw.max() - x_raw.min())
y = (y_raw - y_raw.min())/(y_raw.max() - y_raw.min())

for e in range(epoch):
    y_pred = a * x + b
    grad_a, grad_b = (y_pred - y).dot(x), (y_pred - y).sum()
    a, b = a - lr * grad_a,  b - lr * grad_b
    # plt.scatter(x, y)
    # xx = np.arange(0, 1, 0.01)
    # yy = a * xx + b
    # plt.plot(xx, yy)
    # plt.show()
print(a, b)


