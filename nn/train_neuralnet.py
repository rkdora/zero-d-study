import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from two_layer_net import TwoLayerNet
from keras.utils.np_utils import to_categorical
from common.trainer import Trainer

# データの読み込み
(x_train, t_train), (x_test, t_test) = mnist.load_data()

# 1次元へ整形
x_train, x_test = x_train.reshape(-1, 784), x_test.reshape(-1, 784)

# 正規化
x_train, x_test = x_train.astype(np.float32) / 255.0, x_test.astype(np.float32) / 255.0

# One-hot-vector
t_train, t_test = to_categorical(t_train), to_categorical(t_test)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

max_epochs = 1000

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='SGD', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params_nn.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.title("Acc")
# plt.show()
plt.savefig("acc_nn.png")

plt.cla()

x = np.arange(len(trainer.train_loss_list))
plt.plot(x, trainer.train_loss_list, marker='o', label='train')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 10)
plt.legend(loc='lower right')
plt.title("Loss")
# plt.show()
plt.savefig("loss_nn.png")
