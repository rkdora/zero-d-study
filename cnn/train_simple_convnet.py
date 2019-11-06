import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

# データの読み込み
(x_train, t_train), (x_test, t_test) = mnist.load_data()

print("x_train.shape", x_train.shape)

x_train_shape = x_train.shape
x_train = x_train.reshape(x_train_shape[0], 1, x_train_shape[1], x_train_shape[2])

x_test_shape = x_test.shape
x_test = x_test.reshape(x_test_shape[0], 1, x_test_shape[1], x_test_shape[2])

print("x_train.shape", x_train.shape)

max_epochs = 20

network = SimpleConvNet(input_dim=(1,28,28),
                        conv_param = {'filter_num': 30, 'filter_size': 3, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='SGD', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000)
trainer.train()

# パラメータの保存
network.save_params("params_20.pkl")
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
plt.savefig("acc_20.png")

plt.cla()

x = np.arange(len(trainer.train_loss_list))
plt.plot(x, trainer.train_loss_list, marker='o', label='train')
plt.xlabel("epochs")
plt.ylabel("loss")
plt.ylim(0, 10)
plt.legend(loc='lower right')
plt.title("Loss")
# plt.show()
plt.savefig("loss_20.png")
