import numpy as np
from keras.datasets import mnist
from two_layer_net import TwoLayerNet
import matplotlib.pyplot as plt

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

train_loss_list = []
train_acc_list = []
test_acc_list = []

iters_num = 10000
train_size = X_train.shape[0]
batch_size = 100
learning_rate = 0.1

iter_per_epoch = max(train_size / batch_size, 1)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    # ミニバッチの取得
    batch_mask = np.random.choice(train_size, batch_size)
    X_batch = X_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 勾配の計算
    # grad = network.numerical_gradient(X_batch, t_batch)
    grad = network.gradient(X_batch, t_batch)

    # パラメータの更新
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(X_batch, t_batch)
    train_loss_list.append(loss)

    # 1epoch毎に認識精度の計算
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(X_train, t_train)
        test_acc = network.accuracy(X_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# グラフの描画
# 損失関数
x = np.arange(len(train_loss_list))
plt.plot(x, train_loss_list, label='train loss')
plt.xlabel("epochs")
plt.xlim(0, len(train_loss_list))
plt.ylabel("loss")
plt.ylim(0, 10)
plt.legend(loc='lower right')
plt.title("loss")
# plt.show()
plt.savefig('loss.png')

plt.clf()

# 認識精度
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("accuracy")
# plt.show()
plt.savefig('accuracy.png')
