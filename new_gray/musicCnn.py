
# coding: utf-8

# In[1]:

import tensorflow as tf
import csv
import numpy as np
from PIL import Image


# In[2]:

trainXData = open('trainX.csv','r')
trainYData = open('trainY.csv','r')
testXData = open('testX.csv','r')
testYData = open('testY.csv','r')
trainYDict = np.zeros(8,dtype = np.int64)
testYDict = np.zeros(8,dtype = np.int64)
testX = []
testY = []
trainX = []
trainY = []
trainX_queue = []
testX_queue = []
ansDict=dict()
num = 0
countA = 0


# In[3]:
for i in csv.reader(testXData):
    testX.append(i)
for i in csv.reader(testYData):
    testY.append(i)
for i in csv.reader(trainXData):
    trainX.append(i)
for i in csv.reader(trainYData):
    trainY.append(i)


# In[4]:

for i in trainX:
    trainX_queue.append((i[0]))
for i in testX:
    testX_queue.append((i[0]))

# In[5]:

#filename_queue = tf.train.string_input_producer(trainX_queue)
#reader = tf.WholeFileReader()
#filename, content = reader.read(filename_queue)
##image = tf.image.decode_jpeg(content, channels=3)
#image = tf.cast(image, tf.float32)
#trainX_image = tf.image.resize_images(image, 28, 28)


# In[14]:

train_images = []
for i in trainX_queue:
    image = Image.open(i)
    image = image.resize((100,100))
    train_images.append(np.array(image))
test_images = []
for i in testX_queue:
    image = Image.open(i)
    image = image.resize((100,100))
    test_images.append(np.array(image))
#train_images = np.array(train_images)


# In[15]:

train_image = []
for i in train_images:
    train_image.append(i.flatten())
test_image = []
for i in test_images:
    test_image.append(i.flatten())
    #print(len(i.flatten()))
    #break


# In[8]:

for i in range(8):
    ansDict[num] = np.zeros(8,dtype=np.int64)
    ansDict[num][countA] = 1
    countA +=1
    num=round(num + 1,2)


# In[9]:

for i in trainY:
    trainYDict = np.row_stack((trainYDict,ansDict[int(i[0])]))
trainYDict = np.delete(trainYDict,0,0)

for i in testY:
    testYDict = np.row_stack((testYDict,ansDict[int(i[0])]))
testYDict = np.delete(testYDict,0,0)

# In[10]:

#image_batch = tf.train.batch([trainX_image], batch_size=8)
#trainY_batch = tf.train.batch([trainYDict], batch_size=8)
#train_image = np.asarray(trainX_image)


# In[21]:



def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 100*100]) # 28x28
ys = tf.placeholder(tf.float32, [None, 8])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 100, 100, 1])
# print(x_image.shape)  # [n_samples, 28,28,1]

## conv1 layer ##
W_conv1 = weight_variable([5,5, 1,32]) # patch 5x5, in size 1, out size 32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)                                         # output size 14x14x32

## conv2 layer ##
W_conv2 = weight_variable([5,5, 32, 16]) # patch 5x5, in size 32, out size 64
b_conv2 = bias_variable([16])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)                                         # output size 7x7x64

## func1 layer ##
W_fc1 = weight_variable([25*25*16, 1024])
b_fc1 = bias_variable([1024])
# [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
h_pool2_flat = tf.reshape(h_pool2, [-1, 25*25*16])
h_fc1 = tf.nn.sigmoid(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

## func2 layer ##
W_fc2 = weight_variable([1024, 8])
b_fc2 = bias_variable([8])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(tf.clip_by_value(prediction, 1e-10,1.0)),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()# important step
sess.run(tf.initialize_all_variables())


# In[12]:

"""
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
  # Start populating the filename queue.
    coord = tf.train.Coordinator()
    print(coord)
    threads = tf.train.start_queue_runners(sess=sess,coord=coord)
    for i in range(1): #length of your filename list
        ima = image.eval() #here is your image Tensor :)
        print((ima))
        #Image.fromarray(np.asarray(ima)).show()
    coord.request_stop()
    coord.join(threads)
"""


# In[22]:
for i in range(1000):
    sess.run(train_step, feed_dict={xs: train_image, ys: trainYDict, keep_prob: 0.5})
    if i % 2 == 0:
        train_loss = sess.run(cross_entropy, feed_dict={xs: train_image, ys: trainYDict, keep_prob: 1})
        print("train_loss: ",train_loss)
        print(i,compute_accuracy(test_image,testYDict))
