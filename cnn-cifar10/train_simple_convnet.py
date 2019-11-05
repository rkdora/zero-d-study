import numpy as np
import matplotlib.pyplot as plt
from simple_convnet import SimpleConvNet
from common.trainer import Trainer
from keras.datasets import cifar10

# データの読み込み
(x_train, t_train), (x_test, t_test) = cifar10.load_data()

x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)

x_test = x_test.reshape(x_test.shape[0], 3, 32, 32)

# 正規化
x_train = x_train.astype(np.float32) / 255.0

x_test = x_test.astype(np.float32) / 255.0

x_train = x_train[:5000]
t_train = t_train[:5000]
x_test = x_test[:1000]
t_test = t_test[:1000]

max_epochs = 10

network = SimpleConvNet(input_dim=(3,32,32),
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='SGD', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params.pkl")
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
# plt.show()
plt.savefig("graph.png")
