import tensorflow as tf
import gym
import numpy as np

env = gym.make('DartBlockPush-v0')

mb_obs = []
obs = env.reset()
action = np.random.uniform(-1, 1, 2)
mb_obs.append(np.concatenate([obs['observation'], action * 0]))
def predict(obs_seq):
    # sess = tf.get_default_session()
    sess = tf.get_default_session()
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(
        '/home/niranjan/Projects/vis_inst/baselines/baselines/ppo2/mass_predict-20000.meta')
    saver.restore(sess, tf.train.latest_checkpoint('/home/niranjan/Projects/vis_inst/baselines/baselines/ppo2/'))

    # Now, let's access and create placeholders variables and
    # create feed-dict to feed new data

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    op_to_restore = graph.get_tensor_by_name("predict:0")
    feed_dict = {x: np.expand_dims(obs_seq, axis=0)}
    # print(sess.run(op_to_restore, feed_dict))
    return sess.run(op_to_restore, feed_dict)

# with tf.Session() as sess:
#     for i in range(60):
#         action = np.array([0.5,0.5])#np.random.uniform(-1, 1, 2)
#         obs, _, done, _ = env.step(np.concatenate([action, np.random.uniform(0.5, 4, 1)]))
#
#         mb_obs.append(np.concatenate([obs['observation'], action]))
#         if (i%3==1):
#             print(obs['mass'])
#             obs_seq = np.array(mb_obs[-2:])
#             print(predict(obs_seq))
#
#         if (done):
#             obs = env.reset()
#             action = np.random.uniform(-1, 1, 2)
#             mb_obs = []
#

# Now, access the op that you want to run.


# This will print 60 which is calculated
# using new values of w1 and w2 and saved value of b1.
