# Convolutional neural network with 9 layers (6 conv + 3 fc).
#
# Input data is image 192*192 RGB = 192*192*3 = 110592 neurons.
# Output data is matrix of 3 elements. It is probability of location objects on the image.
# Example of the code is taken of Martin Goerner for MNIST data and edit by me for recognizing aerial photos. [DB]
# https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd

# The model is:
# neural network structure for this sample:
#
# · · · · · · · · · ·      (input data, 3-deep)                     X [batch, 192, 192, 3]
# @ @ @ @ @ @ @ @ @ @   -- conv. layer +BN 7x7x3=>147 stride 1      W1 [7, 7, 3, 12]             B1 [12]
# ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                               Y1 [batch, 192, 192, 12]
#   @ @ @ @ @ @ @ @     -- conv. layer +BN 5x5x12=>300 stride 2     W2 [5, 5, 12, 24]            B2 [24]
#   ∶∶∶∶∶∶∶∶∶∶∶∶∶∶∶                                                 Y2 [batch, 96, 96, 24]
#     @ @ @ @ @ @       -- conv. layer +BN 4x4x24=>384 stride 2     W3 [4, 4, 24, 48]            B3 [48]
#     ∶∶∶∶∶∶∶∶∶∶∶                                                   Y3 [batch, 48, 48, 48]
#      @ @ @ @ @        -- conv. layer +BN 3x3x48=>432 stride 2     W4 [3, 3, 48, 64]            B4 [64]
#      ∶∶∶∶∶∶∶∶∶                                                    Y4 [batch, 24, 24, 64]
#       @ @ @ @         -- conv. layer +BN 3x3x64=>576 stride 2     W5 [3, 3, 64, 96]            B5 [96]
#       ∶∶∶∶∶∶∶                                                     Y5 [batch, 12, 12, 96]
#        @ @ @          -- conv. layer +BN 3x3x96=>864 stride 2     W6 [3, 3, 96, 128]           B6 [128]
#        ∶∶∶∶∶                                                      Y6 [batch, 6, 6, 128] => reshaped to YY [batch, 6*6*128]
#      \x/x\x\x/ ✞      -- fully connected layer (relu+dropout+BN)  W7 [6*6*128, 512]            B7 [512]
#       · · · ·                                                     Y7 [batch, 512]
#       \x/x\x/ ✞       -- fully connected layer (relu+dropout+BN)  W8 [512, 64]                 B8 [64]
#        · · ·                                                      Y8 [batch, 64]
#         \x/           -- fully connected layer (softmax)          W9 [64, 3]                   B9 [3]
#          ·                                                        Y [batch, 3]

from Prepare_data import DataSetGenerator # Batch Creation
import matplotlib
import numpy as np
matplotlib.use('Agg') # For using in Google Cloud Platform
import tensorflow as tf
import matplotlib.pyplot as plt # Plot Accuracy and Cost
import time # Measuring time of training
import os
import math

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0) # 0 used to initialize a pseudorandom number generator

# -----------------------
# Parameters for changing:
aa, cc, b, w = [], [], [], [] # for saving accuracy (a, aa - is the summing), cross_entropy (c, c - is the summing)
# b - bias, w = weights
a, c = 0, 0 # initial value

pred_cycles = 959 # initial value
train_result = 6 # after epochs show result in display
batch_size = 90
Batch_Class = DataSetGenerator("./train")
Batch_Class_Test = DataSetGenerator("./test")
data_train_size = 2880
# 1128*3=3384; 3384* 85% = 2880 for train and 15% = 504 for test; also 2880 or 504 % 3 = 0; 2880 or 504 % 15 = 0
data_test_size = 504
epochs = 10

max_learning_rate = 0.01
min_learning_rate = 0.001
decay_speed = 4
# Большие значения (0,01 – 1) будут соответствовать большому значению шага коррекции. При этом алгоритм будет работать быстрее 
# (т.е. для поиска минимума функции ошибки потребуется меньше шагов), однако  снижается точность настройки на минимум, что 
# потенциально увеличит ошибку обучения. (Покажу график с 1600 и 0,02-0,0001 как он прыгает)
#
# Малые значения коэффициента (0,0001 – 0,001) соответствуют меньшему шагу коррекции весов. При этом число шагов (или эпох), 
# требуемое для поиска оптимума, как правило, увеличивается, но возрастает и точность настройки на минимум, что потенциально 
# уменьшает ошибку обучения. На практике коэффициент скорости обучения обычно подбирают экспериментально.
# При малых значениях я и так имею 100% точность и мне не требуется длительное время обучать по сути дела бесполезно 
# сеть (перфекционизм)
#
# speed экспериментально по графику, чтобы уменьшалась скорость "равноразрядно", т.е. равномерно в пределах одного 
# десятичного разряда и минимальное значение приближалось к заданному 0,001
# End for changing
# -----------------------

# six convolutional layers with their channel counts, and a
# fully connected layer (tha last layer has 3 softmax neurons)
K = 12  # 1st convolutional layer output depth
L = 24  # 2nd convolutional layer output depth
M = 48  # 3rd convolutional layer
N = 64  # 4th convolutional layer
O = 96  # 5th convolutional layer
P = 128 # 6th convolutional layer
Q = 512 # 1st fully connected layer
R = 64  # 2nd fully connected layer
# _ = 3 # 3rd fully connected layer

#epochs = (pred_cycles - 1) * batch_size // data_train_size + 1
cycles = epochs * data_train_size / batch_size
Listing = '' # Show information about training
# min_learning_rate_exp = min_learning_rate / 1.1 # Last learning rate should be even the same with min_learning rate, no more
# learning rate start with value of max_learning rate and last with value of min_learning rate
# decay_speed = - cycles / math.log((min_learning_rate - min_learning_rate_exp) / (max_learning_rate - min_learning_rate_exp))
decay_speed = int(decay_speed)
speed = (min_learning_rate, max_learning_rate, decay_speed)

print('\nSpeed parameters (min, max, speed):', speed)
Listing += '\nSpeed parameters (min, max, speed):' + str(speed)
print('\nTotal Number of Iterations:', int(cycles))
Listing += '\nTotal Number of Iterations:' + str(int(cycles))
print('\nTotal Number of Epochs:', epochs, '\n')
Listing += '\nTotal Number of Epochs:' + str(epochs) + '\n'


def batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    exp_moving_avg = tf.train.ExponentialMovingAverage(0.999, iteration) # adding the iteration prevents from averaging across non-existing iterations
    bnepsilon = 1e-5
    if convolutional:
        mean, variance = tf.nn.moments(Ylogits, [0, 1, 2])
    else:
        mean, variance = tf.nn.moments(Ylogits, [0])
    update_moving_averages = exp_moving_avg.apply([mean, variance])
    m = tf.cond(is_test, lambda: exp_moving_avg.average(mean), lambda: mean)
    v = tf.cond(is_test, lambda: exp_moving_avg.average(variance), lambda: variance)
    Ybn = tf.nn.batch_normalization(Ylogits, m, v, offset, None, bnepsilon)
    return Ybn, update_moving_averages

def no_batchnorm(Ylogits, is_test, iteration, offset, convolutional=False):
    return Ylogits, tf.no_op()

def compatible_convolutional_noise_shape(Y):
    noiseshape = tf.shape(Y)
    noiseshape = noiseshape * tf.constant([1,0,0,1]) + tf.constant([0,1,1,0])
    return noiseshape

# input X: 192x192x3 color images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 192, 192, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 3])
# variable learning rate
lr = tf.placeholder(tf.float32)
# test flag for batch norm
tst = tf.placeholder(tf.bool)
iter = tf.placeholder(tf.int32)
# dropout probability
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)

# Convolutional layers
W1 = tf.Variable(tf.truncated_normal([7, 7, 3, K], stddev=0.1))  # 7x7 patch, 3 input channel, K output channels
B1 = tf.Variable(tf.constant(0.1, tf.float32, [K]))
W2 = tf.Variable(tf.truncated_normal([5, 5, K, L], stddev=0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, [L]))
W3 = tf.Variable(tf.truncated_normal([4, 4, L, M], stddev=0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, [M]))
W4 = tf.Variable(tf.truncated_normal([3, 3, M, N], stddev=0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, [N]))
W5 = tf.Variable(tf.truncated_normal([3, 3, N, O], stddev=0.1))
B5 = tf.Variable(tf.constant(0.1, tf.float32, [O]))
W6 = tf.Variable(tf.truncated_normal([3, 3, O, P], stddev=0.1))
B6 = tf.Variable(tf.constant(0.1, tf.float32, [P]))
# Fully-connected layers
W7 = tf.Variable(tf.truncated_normal([6 * 6 * P, Q], stddev=0.1))
B7 = tf.Variable(tf.constant(0.1, tf.float32, [Q]))
W8 = tf.Variable(tf.truncated_normal([Q, R], stddev=0.1))
B8 = tf.Variable(tf.constant(0.1, tf.float32, [R]))
W9 = tf.Variable(tf.truncated_normal([R, 3], stddev=0.1))
B9 = tf.Variable(tf.constant(0.1, tf.float32, [3]))

# The model
# batch norm scaling is not useful with relus
# batch norm offsets are used instead of biases
stride = 1  # output is 192x192
Y1l = tf.nn.conv2d(X, W1, strides=[1, stride, stride, 1], padding='SAME')
Y1bn, update_ema1 = batchnorm(Y1l, tst, iter, B1, convolutional=True)
Y1r = tf.nn.relu(Y1bn)
Y1 = tf.nn.dropout(Y1r, pkeep_conv, compatible_convolutional_noise_shape(Y1r))
stride = 2  # output is 96x96
Y2l = tf.nn.conv2d(Y1, W2, strides=[1, stride, stride, 1], padding='SAME')
Y2bn, update_ema2 = batchnorm(Y2l, tst, iter, B2, convolutional=True)
Y2r = tf.nn.relu(Y2bn)
Y2 = tf.nn.dropout(Y2r, pkeep_conv, compatible_convolutional_noise_shape(Y2r))
stride = 2  # output is 48x48
Y3l = tf.nn.conv2d(Y2, W3, strides=[1, stride, stride, 1], padding='SAME')
Y3bn, update_ema3 = batchnorm(Y3l, tst, iter, B3, convolutional=True)
Y3r = tf.nn.relu(Y3bn)
Y3 = tf.nn.dropout(Y3r, pkeep_conv, compatible_convolutional_noise_shape(Y3r))
stride = 2  # output is 24x24
Y4l = tf.nn.conv2d(Y3, W4, strides=[1, stride, stride, 1], padding='SAME')
Y4bn, update_ema4 = batchnorm(Y4l, tst, iter, B4, convolutional=True)
Y4r = tf.nn.relu(Y4bn)
Y4 = tf.nn.dropout(Y4r, pkeep_conv, compatible_convolutional_noise_shape(Y4r))
stride = 2  # output is 12x12
Y5l = tf.nn.conv2d(Y4, W5, strides=[1, stride, stride, 1], padding='SAME')
Y5bn, update_ema5 = batchnorm(Y5l, tst, iter, B5, convolutional=True)
Y5r = tf.nn.relu(Y5bn)
Y5 = tf.nn.dropout(Y5r, pkeep_conv, compatible_convolutional_noise_shape(Y5r))
stride = 2  # output is 6x6
Y6l = tf.nn.conv2d(Y5, W6, strides=[1, stride, stride, 1], padding='SAME')
Y6bn, update_ema6 = batchnorm(Y6l, tst, iter, B6, convolutional=True)
Y6r = tf.nn.relu(Y6bn)
Y6 = tf.nn.dropout(Y6r, pkeep_conv, compatible_convolutional_noise_shape(Y6r))

# reshape the output from the sixth convolution for the fully connected layer
YY = tf.reshape(Y6, shape=[-1, 6 * 6 * P])

Y7l = tf.matmul(YY, W7)
Y7bn, update_ema7 = batchnorm(Y7l, tst, iter, B7)
Y7r = tf.nn.relu(Y7bn)
Y7 = tf.nn.dropout(Y7r, pkeep)
Y8l = tf.matmul(Y7, W8)
Y8bn, update_ema8 = batchnorm(Y8l, tst, iter, B8)
Y8r = tf.nn.relu(Y8bn)
Y8 = tf.nn.dropout(Y8r, pkeep)

Ylogits = tf.matmul(Y8, W9) + B9
Y = tf.nn.softmax(Ylogits)

update_ema = tf.group(update_ema1, update_ema2, update_ema3, update_ema4, update_ema5, 
                      update_ema6, update_ema7, update_ema8)

# TensorFlow provides the softmax_cross_entropy_with_logits function to avoid numerical stability
# problems with log(0) which is NaN
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=Ylogits, labels=Y_)
cross_entropy = tf.reduce_mean(cross_entropy)*100

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

W = tf.concat([tf.reshape(W1, [-1]), tf.reshape(W2, [-1]), tf.reshape(W3, [-1]), tf.reshape(W4, [-1]),
    tf.reshape(W5, [-1]), tf.reshape(W6, [-1]), tf.reshape(W7, [-1]), tf.reshape(W8, [-1]), tf.reshape(W9, [-1])], 0)
B  = tf.concat([tf.reshape(B1, [-1]), tf.reshape(B2, [-1]), tf.reshape(B3, [-1]), tf.reshape(B4, [-1]),
    tf.reshape(B5, [-1]), tf.reshape(B6, [-1]), tf.reshape(B7, [-1]), tf.reshape(B8, [-1]), tf.reshape(B9, [-1])], 0)

# training step, the learning rate is a placeholder
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

start_time = time.time()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
i = 0

for epoch in range(epochs):
    batches = Batch_Class.get_mini_batches(batch_size)
    # learning rate decay

    learning_rate = min_learning_rate + (max_learning_rate - min_learning_rate) * math.exp(-epoch / decay_speed) #_exp

    for batch_X, batch_Y in batches:
        batch_X = np.divide(batch_X, 255)
        i += 1

        # the backpropagation training step
        sess.run(train_step, {X: batch_X, Y_: batch_Y, lr: learning_rate, tst: False, pkeep: 0.75, pkeep_conv: 1.0})
        sess.run(update_ema, {X: batch_X, Y_: batch_Y, tst: False, iter: i, pkeep: 1.0, pkeep_conv: 1.0})

        update_train_data = i % train_result == 0
        if update_train_data: 
            a, c = sess.run([accuracy, cross_entropy], {X: batch_X, Y_: batch_Y, tst: False, pkeep: 1.0, pkeep_conv: 1.0})
            print(str(i) + ": accuracy:" + str(round(a, 4)) + " loss: " + str(round(c, 4)))
            Listing += '\n' + str(i) + ": accuracy:" + str(round(a, 4)) + " loss: " + str(round(c, 4))

    test_batch = Batch_Class_Test.get_test_batch() #batch_test_X, batch_test_Y
    batch_test_X, batch_test_Y = next(test_batch)
    batch_test_X = np.divide(batch_test_X, 255)
    a, c = sess.run([accuracy, cross_entropy], {X: batch_test_X, Y_: batch_test_Y, 
                                                            tst: True, pkeep: 1.0, pkeep_conv: 1.0})
    print(str(i) + ": ********* epoch " + str(epoch + 1) + " ********* test accuracy:" +
          str(round(a, 4)) + " test loss: " + str(round(c, 4)) + ' learning rate:' + str(round(learning_rate, 5)))
    Listing += '\n' + str(i) + ": ********* epoch " + str(epoch + 1) + " ********* test accuracy:" + str(round(a, 4)) + " test loss: " + str(round(c, 4)) + ' learning rate:' + str(round(learning_rate, 5))
    aa.append(a)
    cc.append(c)

a, c, w1, w2, w3, w4, w5, w6, w7, w8, w9, b = sess.run([accuracy, cross_entropy, W1, W2, W3, W4, W5, W6, W7, W8, W9, B],
                                          {X: batch_test_X, Y_: batch_test_Y, tst: True, pkeep: 1.0, pkeep_conv: 1.0})
print(str(i) + ": ********* epoch " + str(epoch + 1) +
               " ********* test accuracy:" +str(round(a, 4)) + " test loss: " + str(round(c, 4)))
Listing +=  '\n' + str(i) + ": ********* epoch " + str(epoch + 1) + " ********* test accuracy:" + str(round(a, 4)) + " test loss: " + str(round(c, 4)) + ' learning rate:' + str(round(learning_rate, 5))


del batch_X
del batch_Y
del batch_test_X
del batch_test_Y

training_duration = (time.time() - start_time)
print("\n-— Training finished after %d seconds —- \n" % training_duration )
print('Shape of Biases:', b.size)
print('\nShape of Weights1:', w1.shape)
print('Shape of Weights2:', w2.shape)
print('Shape of Weights3:', w3.shape)
print('Shape of Weights4:', w4.shape)
print('Shape of Weights5:', w5.shape)
print('Shape of Weights6:', w6.shape)
print('Shape of Weights7:', w7.shape)
print('Shape of Weights8:', w8.shape)
print('Shape of Weights9:', w9.shape)
print("\nMax Test Accuracy: " + str(max(aa)))

info = "Training finished after %d seconds" % training_duration
info += "\nMax Test Accuracy:" + str(max(aa))
info += '\nTotal Number of Epochs:' + str(epochs)
info += "\nTensorflow version " + tf.__version__
info += '------------------------------------------------'
info += '\nListing:\n'
info += Listing

str_cycles = str(int(cycles))
str_speed = str(speed)
str_batch_size = str(int(batch_size))
info_name = 'speed-' + str_speed + '.cycles-' + str_cycles + '.batchsize-' + str_batch_size
os.makedirs(info_name)

np.set_printoptions(threshold=np.nan)

plt.figure()
x = list(range(0, int(cycles), int(data_train_size / batch_size)))
plt.plot(x, aa, 'b-')
plt.savefig(info_name + '/CNN42_a.' + info_name + '.png', format='png', dpi=600)

plt.figure()
plt.plot(x, cc, 'r-')
plt.savefig(info_name + '/CNN42_c.' + info_name + '.png', format='png', dpi=600)

plt.figure()
plt.plot(x, cc, 'r-', aa, 'b-')
plt.savefig(info_name + '/CNN42_ac.' + info_name + '.png', format='png', dpi=600)


def save_txt(name, parameter):
    text_file = open(info_name + name + info_name + '.txt', 'w', encoding='utf-8')
    text_file.write(str(parameter))
    text_file.close()
    del parameter

save_txt('/CNN42_b1.', b)
save_txt('/CNN42_w1.', w1)
save_txt('/CNN42_w2.', w2)
save_txt('/CNN42_w3.', w3)
save_txt('/CNN42_w4.', w4)
save_txt('/CNN42_w5.', w5)
save_txt('/CNN42_w6.', w6)
save_txt('/CNN42_w7.', w7)
save_txt('/CNN42_w8.', w8)
save_txt('/CNN42_w9.', w9)
save_txt('/CNN42_info.', info)
save_txt('/CNN42_a.', aa)
save_txt('/CNN42_c.', cc)
sess.close()