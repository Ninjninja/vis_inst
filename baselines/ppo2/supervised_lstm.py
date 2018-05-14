import tensorflow as tf
from tensorflow.contrib import rnn
import tensorboard
import numpy as np
# import mnist dataset
from tensorflow.examples.tutorials.mnist import input_data
import gym

env = gym.make('DartBlockPush-v0')
# env.reset()
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# define constants
# unrolled through 28 time steps
time_steps = 2
# hidden LSTM units
num_units = 128
# rows of 28 pixels
n_input = 5
# learning rate for adam
learning_rate = 0.01
# mnist is meant to be classified in 10 classes(0-9).
n_classes = 1
# size of batch
batch_size = 128

x = tf.placeholder(tf.float32, [None, time_steps, n_input], name='x')
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


# prediction=tf.matmul(outputs[-1],out_weights)+out_bias
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
prediction = simple_fc_network(x)
loss = tf.reduce_mean(tf.losses.mean_squared_error(prediction, y))
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss, name='min_loss')

# correct prediction
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
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
    mb_obs, mb_action, y1, x1 = [], [], [], []
    obs = env.reset()
    obs = env.reset()
    mb_obs.append(obs['observation'])
    for i in range(30000):
        action = np.random.uniform(-1, 1, 2)
        obs, _, done, _ = env.step(np.concatenate([action, np.random.uniform(0.5, 4, 1)]))
        mb_obs.append(np.concatenate([obs['observation'], action]))
        mb_action.append(action)
        if (done):
            y1.append([obs['mass']])
            x1.append(np.array(mb_obs[:-1]))
            obs = env.reset()
            mb_obs.append(np.concatenate([obs['observation'], action * 0]))
            mb_obs = []
    X_train = np.array(x1[1:8000])
    Y_train = np.array(y1[1:8000])
    X_test = np.array(x1[8000:])
    Y_test = np.array(y1[8000:])
    np.save('X_train', X_train)
    np.save('Y_train', Y_train)
    for i in range(20000):
        sess.run(opt, feed_dict={x: X_train, y: Y_train})
        if i % 10 == 0:
            # acc = sess.run(accuracy, feed_dict={x: X, y: Y})
            los, summary_out = sess.run([loss, merged], feed_dict={x: X_test, y: Y_test})
            train_writer.add_summary(summary_out, i)
            print("For iter ", iter)
            # print("Accuracy ", acc)
            print("Loss ", los)
            print("__________________")

    saver.save(sess, './mass_predict', global_step=20000)
