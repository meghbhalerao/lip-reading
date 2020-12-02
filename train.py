import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
# 50 utterances
visual_mouth_data = np.random.rand(25000,9,60,100,1)
audio_spectogram_data = np.random.rand(15,40,3,25000)

common_labels = np.zeros([500,1])
test = np.eye(500)
for i in range(500):
  temp = test[:,[i]]
  #print(np.shape(temp))
  print(np.shape(common_labels))  
  for j in range(50):
    common_labels = np.append(common_labels,temp,axis = 1)
    
print(np.shape(temp))
print(np.shape(common_labels))
common_labels = common_labels[:,1:25001]#.transpose()
print(np.shape(common_labels))

x_input = tf.placeholder(tf.float32, shape = [1,9,60,100,1])
y_input = tf.placeholder(tf.float32, shape = [500,1])
#common_labels = commn_labels.reshape(1,500,25000)
#visual_mouth_data = visual_mouth_data.reshape()

def visual_CNN(input):
    #Layer1
    conv1 = tf.layers.conv3d(input, filters = 16, kernel_size = [3,3,3], strides=[ 1, 1, 1], padding="SAME",activation = tf.nn.leaky_relu, use_bias = True)
    conv1 = tf.layers.max_pooling3d(conv1,pool_size = [1,3,3],strides = [1,2,2])

    conv2 = tf.layers.conv3d(conv1, filters= 32, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="SAME",activation=tf.nn.leaky_relu, use_bias=True)
    conv2 = tf.layers.max_pooling3d(conv2, pool_size=[1, 3, 3], strides=[1, 2, 2])

    conv3 = tf.layers.conv3d(conv2, filters=64, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="SAME", activation=tf.nn.leaky_relu, use_bias=True)
    conv3 = tf.layers.max_pooling3d(conv3, pool_size=[1, 3, 3], strides=[1, 2, 2])

    conv4 = tf.layers.conv3d(conv3, filters=128, kernel_size=[3, 3, 3], strides=[1, 1, 1], padding="SAME", activation=tf.nn.leaky_relu, use_bias=True)

    flattening = tf.reshape(conv4,[-1,1])
    
    dense1 = tf.contrib.layers.fully_connected(flattening,1000,activation_fn = tf.nn.leaky_relu)
    #dropout = tf.layers.dropout(inputs = dense1,rate = keep_rate,training = True)
    dense2 = tf.contrib.layers.fully_connected(dense1,500,activation_fn = tf.nn.leaky_relu)
    
    
    return dense2


def train(x_train,y_train,learning_rate = 0.05, epochs = 10):
    prediction = visual_CNN(x_input)
    cost = tf.losses.mean_squared_error(y_input,prediction)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    iterations = len(x_train)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        iterations = len(x_train)

        for epoch in range(epochs):
            print("Epochs", epoch, "started")
            epoch_loss = 0

            for itr in range(iterations):
                  temp = x_train[itr,:,:,:,:] 
                  temp.reshape(1,9,60,100,1)
                  _optimizer, _cost = sess.run([optimizer,cost],feed_dict = {x_input : temp, y_input :y_train[:,itr]})

train(visual_mouth_data,common_labels,learning_rate = 0.05,epochs = 10)
