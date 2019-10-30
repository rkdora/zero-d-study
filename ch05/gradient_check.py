import numpy as np
from keras.datasets import mnist
from two_layer_net import TwoLayerNet

(X_train, t_train), (X_test, t_test) = mnist.load_data()

# 二次元から一次元へ
X_train = np.array(list(map(lambda x: x.reshape(-1, ), X_train)))
X_test = np.array(list(map(lambda x: x.reshape(-1, ), X_test)))

# one-hot-vector
t_train = np.identity(10)[t_train]
t_test = np.identity(10)[t_test]

print("X_train.shape", X_train.shape)
print("t_train.shape", t_train.shape)
print("X_test.shape", X_test.shape)
print("t_test.shape", t_test.shape)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

X_batch = X_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(X_batch, t_batch)
grad_backprop = network.gradient(X_batch, t_batch)

# 各重みの絶対誤差の平均を求める
for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ":" + str(diff))
    
