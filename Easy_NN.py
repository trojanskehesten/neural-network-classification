# Easy neural network with 1 layer of 3 softmax neurons
#
# Input data is image 192*192 RGB = 192*192*3 = 110592 neurons
# Output data is matrix of 3 elements. It is probability of location objects on the image
# Example of the code is taken of Martin Goerner for MNIST data and edit by me for recognizing aerial photos. [DB]
# https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd

# The model is:
# · · · · · · · · · ·       (input data, flattened pixels)       X [batch, 110592]
# \x/x\x/x\x/x\x/x\x/    -- fully connected layer (softmax)      W [110592, 3]; b[3]
#   · · · · · · · ·                                              Y [batch, 3]
# Y = softmax( X * W + b)
#              X: matrix for 15 color images of 192x192 pixels, flattened (there are 15 images in a mini-batch)
#              W: weight matrix with 110592 lines and 3 columns
#              b: bias vector with 3 dimensions
#              +: add with broadcasting: adds the vector to each line of the matrix (numpy)
#              softmax(matrix) applies softmax on each line
#              softmax(line) applies an exp to each value then divides by the norm of the resulting line
#              Y: output matrix with 15 lines and 3 columns

from Prepare_data import DataSetGenerator # Batch Creation
import matplotlib
import numpy as np
matplotlib.use('Agg') # For using in Google Cloud Platform
import tensorflow as tf
import matplotlib.pyplot as plt # Plot Accuracy and Cost
import time # Measuring time of training

print("Tensorflow version " + tf.__version__)
tf.set_random_seed(0) # 0 used to initialize a pseudorandom number generator

# -----------------------
# Parameters for changing:

aa, cc, b, w = [], [], [], [] # for saving accuracy (a, aa - is the summing), cross_entropy (c, c - is the summing)
# b - bias, w = weights
a, c = 0, 0 # initial value

pred_cycles = 15000 # initial value
train_result = 24 # after epochs show result in display
batch_size = 15
Batch_Class = DataSetGenerator("./train")
Batch_Class_Test = DataSetGenerator("./test")
data_train_size = 2880
# 1128*3=3384; 3384* 85% = 2880 for train and 15% = 504 for test; also 2880 or 504 % 3 = 0; 2880 or 504 % 15 = 0
data_test_size = 504
# training, learning rate = 0.005
speed = 0.005

# End for changing
# -----------------------

epochs = (pred_cycles - 1) * batch_size // data_train_size + 1
cycles = epochs * data_train_size / batch_size
print('\nTotal Number of Iterations:', int(cycles))
print('\nTotal Number of Epochs:', epochs, '\n')

# input X: 192x192x3 color images, the first dimension (None) will index the images in the mini-batch
X = tf.placeholder(tf.float32, [None, 192, 192, 3])
# correct answers will go here
Y_ = tf.placeholder(tf.float32, [None, 3])
# weights W[110592, 3]   110592=192*192*3
W = tf.Variable(tf.zeros([110592, 3]))
# biases b[3]
B = tf.Variable(tf.zeros([3]))

# flatten the images into a single line of pixels
# -1 in the shape definition means "the only possible dimension that will preserve the number of elements"
XX = tf.reshape(X, [-1, 110592])

# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + B)

# loss function: cross-entropy = - sum( Y_i * log(Yi) )
#                           Y: the computed output vector
#                           Y_: the desired output vector

# cross-entropy
# log takes the log of each element, * multiplies the tensors element by element
# reduce_mean will add all the components in the tensor
# so here we end up with the total cross-entropy for all images in the batch
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y))

# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training, learning rate = speed
train_step = tf.train.GradientDescentOptimizer(speed).minimize(cross_entropy)

start_time = time.time()

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
i = 0

for epoch in range(epochs):
    batches = Batch_Class.get_mini_batches(batch_size)
    for batch_X, batch_Y in batches:
        i += 1
        batch_X = np.divide(batch_X, 255)
        sess.run(train_step, feed_dict={X: batch_X, Y_: batch_Y})
        update_train_data = i % train_result == 0
        if update_train_data:
            a, c = sess.run([accuracy, cross_entropy], feed_dict={X: batch_X, Y_: batch_Y})
            print(str(i) + ": accuracy:" + str(a) + " loss: " + str(c))

    test_batch = Batch_Class_Test.get_test_batch() #batch_test_X, batch_test_Y
    batch_test_X, batch_test_Y = next(test_batch)
    batch_test_X = np.divide(batch_test_X, 255)
    a, c, w, b = sess.run([accuracy, cross_entropy, W, B], feed_dict={X: batch_test_X, Y_: batch_test_Y})
    print(str(i) + ": ********* epoch " + str(epoch + 1) +
                                             " ********* test accuracy:" +str(a) + " test loss: " + str(c))
    aa.append(a)
    cc.append(c)

del batch_X
del batch_Y
del batch_test_X
del batch_test_Y

training_duration = (time.time() - start_time)
print("\n-— Training finished after %d seconds —- \n" % training_duration )
print('Shape of Biases:', b.size)
print('Biases:', b)
print('\nShape of Weights:', w.shape)
print('Weights:', w)

info = "Training finished after %d seconds" % training_duration
info += "\nMax Test Accuracy:" + str(max(aa))
info += '\nTotal Number of Epochs:' + str(epochs)
info += "\nTensorflow version " + tf.__version__

str_cycles = str(int(cycles))
str_speed = str(speed)
str_batch_size = str(int(batch_size))
info_name = 'speed-' + str_speed + '.cycles-' + str_cycles + '.batchsize-' + str_batch_size 
# os.makedirs(info_name)

def save_txt(info_name, name, parameter):
    # save parametr to text file
    name = info_name + name + info_name + '.txt'
    text_file = open(name, 'w', encoding='utf-8')
    text_file.write(str(parameter))
    text_file.close()

save_txt(info_name, 'CNN10_a1', aa)
save_txt(info_name, 'CNN10_c1', cc)
save_txt(info_name, 'CNN10_b1', b)
save_txt(info_name, 'CNN10_w1', w)

# plotting and save graph
plt.figure()
x = list(range(0, int(cycles), int(data_train_size / batch_size)))
plt.plot(x, aa, 'b-')
plt.savefig(info_name + '/CNN10_a1.' + info_name + '.png', format='png', dpi=600)
plt.figure()
plt.plot(x, cc, 'r-')
plt.savefig(info_name + '/CNN10_c1.' + info_name + '.png', format='png', dpi=600)

print("\nMax Test Accuracy: " + str(max(aa)))
