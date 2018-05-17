import tensorflow as tf
from tensorflow.contrib import rnn
import tensorboard
import numpy as np
# import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
import gym
from tensorflow.python import debug as tf_debug

env = gym.make('DartBlockPush-v0')
# env.reset()


# define constants
# unrolled through 28 time steps
time_steps = 2
# hidden LSTM units
num_units = 128
# rows of 28 pixels
n_input = 5
# learning rate for adam
learning_rate = 0.005
# mnist is meant to be classified in 10 classes(0-9).
n_classes = 1
# size of batch
batch_size = 128

x = tf.placeholder(tf.float32, [None, time_steps, 256, 256, 3], name='x')
act = tf.placeholder(tf.float32, [None, time_steps, 2], name='action')
y = tf.placeholder(tf.float32, [None, n_classes])


def simple_fc_network(x):
    out_weights = tf.Variable(tf.random_normal([num_units, n_classes]))
    out_bias = tf.Variable(tf.random_normal([n_classes]))
    input = tf.unstack(x, time_steps, axis=1)
    lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1)
    outputs, _ = rnn.static_rnn(lstm_layer, input, dtype=tf.float32)
    print('out')
    prediction = tf.layers.dense(inputs=outputs[-1], units=1)
    prediction = tf.identity(prediction, name='predict')
    return prediction


def cnn_network(input, actions, is_training):
    with tf.variable_scope('cnn_model'):
        dropout = 0.25
        input1 = tf.reshape(input, shape=[-1,256,256,3])
        actions1 = tf.reshape(actions, shape = [-1,2])
        conv1 = tf.layers.conv2d(input1, 32, 16, activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 8, 8)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 8, activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 4, 4)

        conv3 = tf.layers.conv2d(conv2, 64, 4, activation=tf.nn.relu,kernel_initializer=tf.random_normal_initializer)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv3 = tf.layers.max_pooling2d(conv3, 2, 2)
        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv3)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 100)
        fc2 = tf.layers.dense(actions1, 20,activation=tf.nn.relu)
        features = tf.concat([fc1,fc2], axis=1)
        features = tf.reshape(features,shape=[-1,time_steps,120])
        features = tf.unstack(features, time_steps, axis=1)
        lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias=1,name='lstm_output')
        outputs, _ = rnn.static_rnn(lstm_layer, features, dtype=tf.float32)
        prediction = tf.layers.dense(inputs=outputs[-1], units=1,name='pred_fc')
        # prediction = tf.gather(prediction, [1], axis=)
            # Output layer, class prediction
        # prediction =tf.gather(prediction,[num_units-1],axis=0)

        prediction = tf.identity(prediction, name='predict')
        return prediction
# prediction=tf.matmul(outputs[-1],out_weights)+out_bias
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
prediction = cnn_network(x, act, True)
loss = tf.reduce_mean(tf.losses.mean_squared_error(prediction, y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, name='min_loss')

# correct prediction

tf.summary.scalar('loss', loss)
merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('./train', sess.graph)
    sess.run(init)
    # for i in range(1000):
    #     batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)
    #
    #     batch_x=batch_x.reshape((batch_size,time_steps,n_input))
    #
    #     sess.run(opt, feed_dict={x: batch_x, y: batch_y})
    #     if i %10==0:
    #         acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
    #         los,summary_out=sess.run([loss,merged],feed_dict={x:batch_x,y:batch_y})
    #         train_writer.add_summary(summary_out, i)
    #         print("For iter ",iter)
    #         print("Accuracy ",acc)
    #         print("Loss ",los)
    #         print("__________________")
    mb_obs, mb_action, y1, x1, x1_action = [], [], [], [], []

    obs = env.reset()
    obs = env.reset()

    for i in range(300):
        count = 0
        while count < 50:
            action = np.random.uniform(-1, 1, 2)
            obs, _, done, _ = env.step(np.concatenate([action, np.random.uniform(1, 4, 1)]))
            mb_obs.append(obs['observation'])
            mb_action.append(action)
            if (done):
                count += 1
                y1.append([obs['mass']])
                x1.append(np.array(mb_obs[:-1]))
                x1_action.append(np.array(mb_action[:-1]))
                obs = env.reset()
                mb_obs.append(obs['observation'])
                mb_action.append(action)
                mb_obs = []
                mb_action = []

        X_train = np.array(x1)
        X_train_act = np.array(x1_action)
        Y_train = np.array(y1)
        y1 = []
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        for i in range(200):
            los, _, _ = sess.run([loss, opt, merged], feed_dict={x: X_train, y: Y_train, act:X_train_act})
            print(los)
        x1 = []
        x1_action = []

    X_test = np.array(x1[8000:])
    Y_test = np.array(y1[8000:])
    # np.save('X_train', X_train)
    # np.save('Y_train', Y_train)
    for i in range(20000):
        sess.run(opt, feed_dict={x: X_train, y: Y_train, act:X_train_act})
        if i % 10 == 0:
            # acc = sess.run(accuracy, feed_dict={x: X, y: Y})
            los, summary_out = sess.run([loss, merged], feed_dict={x: X_test, y: Y_test})
            train_writer.add_summary(summary_out, i)
            print("For iter ", iter)
            # print("Accuracy ", acc)
            print("Loss ", los)
            print("__________________")

    saver.save(sess, './mass_predict_cnn', global_step=20000)
