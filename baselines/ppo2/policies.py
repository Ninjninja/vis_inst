import numpy as np
import tensorflow as tf
from baselines.a2c.utils import conv, fc, conv_to_fc, batch_to_seq, seq_to_batch, lstm, lnlstm
from baselines.common.distributions import make_pdtype
from gym import spaces
from tensorflow.python import debug as tf_debug

def nature_cnn(unscaled_images):
    """
    CNN from Nature paper.
    """
    scaled_images = tf.cast(unscaled_images, tf.float32) / 255.
    activ = tf.nn.relu
    h = activ(conv(scaled_images, 'c1', nf=32, rf=8, stride=4, init_scale=np.sqrt(2)))
    h2 = activ(conv(h, 'c2', nf=64, rf=4, stride=2, init_scale=np.sqrt(2)))
    h3 = activ(conv(h2, 'c3', nf=64, rf=3, stride=1, init_scale=np.sqrt(2)))
    h3 = conv_to_fc(h3)
    return activ(fc(h3, 'fc1', nh=512, init_scale=np.sqrt(2)))

class LnLstmPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=256, reuse=False):
        nenv = nbatch // nsteps
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch]) #mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm*2]) #states
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lnlstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, mask):
            return sess.run([a0, v0, snew, neglogp0], {X:ob, S:state, M:mask})

        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class LstmPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, nlstm=50, reuse=False):
        nenv = nbatch // nsteps
        # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        # nh, nw, nc = ob_space.shape
        # ob_shape = (nbatch, nh, nw, nc)
        ob_shape = (nbatch,) + ob_space.shape
        X = tf.placeholder(tf.float32, ob_shape)
        nact = ac_space.shape[0]-1
        # X = tf.placeholder(tf.uint8, ob_shape) #obs
        M = tf.placeholder(tf.float32, [nbatch], name='mask')  # mask (done t-1)
        S = tf.placeholder(tf.float32, [nenv, nlstm * 2], name='state')  # states
        S_pred = tf.placeholder(tf.float32, [nenv, nlstm * 2], name='predict_state')  # states
        with tf.variable_scope("model", reuse=reuse):
            # h = nature_cnn(X)
            h = tf.nn.tanh(fc(X, 'fc1', 20))
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew = lstm(xs, ms, S, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            pi = fc(h5, 'pi', nact)
            vf = fc(h5, 'v', 1)
            logstd = tf.get_variable(name="logstd", shape=[1, nact],
                initializer=tf.zeros_initializer())
        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)
        self.pdtype = make_pdtype(spaces.Box(ac_space.low[0],ac_space.high[0],[nact,]))
        self.pd = self.pdtype.pdfromflat(pdparam)

        v0 = vf[:, 0]
        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)

        with tf.variable_scope('predictor', reuse=reuse):
            # h = nature_cnn(X)
            h = tf.nn.relu(fc(X, 'fc1', 20))
            xs = batch_to_seq(h, nenv, nsteps)
            ms = batch_to_seq(M, nenv, nsteps)
            h5, snew_pred = lstm(xs, ms, S_pred, 'lstm1', nh=nlstm)
            h5 = seq_to_batch(h5)
            # h7 = (fc(a0, 'prediction_fc_action', 10))
            # h6 = tf.concat([h7, h5],axis=1)
            h7 = fc(h5, 'prediction_fc', 256)
            self.prediction = tf.nn.relu(fc(h7, 'prediction_out', 1))

        self.initial_state = np.zeros((nenv, nlstm*2), dtype=np.float32)

        def step(ob, state, predict_state, mask):
            prediction_out, a0_out, v0_out, snew_out, snew_predict_out, neglogp0_out = sess.run(
                [self.prediction, a0, v0, snew, snew_pred, neglogp0],
                {X: ob, S: state, S_pred: predict_state, M: mask})

            return np.concatenate([a0_out, prediction_out], axis=-1), v0_out, snew_out, snew_predict_out, neglogp0_out


        def value(ob, state, mask):
            return sess.run(v0, {X:ob, S:state, M:mask})

        self.X = X
        self.M = M
        self.S = S
        self.S_pred = S_pred
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class CnnPolicy(object):

    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        nh, nw, nc = ob_space.shape
        ob_shape = (nbatch, nh, nw, nc)
        nact = ac_space.n
        X = tf.placeholder(tf.uint8, ob_shape) #obs
        with tf.variable_scope("model", reuse=reuse):
            h = nature_cnn(X)
            pi = fc(h, 'pi', nact, init_scale=0.01)
            vf = fc(h, 'v', 1)[:,0]

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pi)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value

class MlpPolicy(object):
    def __init__(self, sess, ob_space, ac_space, nbatch, nsteps, reuse=False): #pylint: disable=W0613
        ob_shape = (nbatch,) + ob_space.shape
        actdim = ac_space.shape[0]
        X = tf.placeholder(tf.float32, ob_shape, name='Ob') #obs
        with tf.variable_scope("model", reuse=reuse):
            activ = tf.tanh
            h1 = activ(fc(X, 'pi_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'pi_fc2', nh=64, init_scale=np.sqrt(2)))
            pi = fc(h2, 'pi', actdim, init_scale=0.01)
            h1 = activ(fc(X, 'vf_fc1', nh=64, init_scale=np.sqrt(2)))
            h2 = activ(fc(h1, 'vf_fc2', nh=64, init_scale=np.sqrt(2)))
            vf = fc(h2, 'vf', 1)[:,0]
            logstd = tf.get_variable(name="logstd", shape=[1, actdim],
                initializer=tf.zeros_initializer())

        pdparam = tf.concat([pi, pi * 0.0 + logstd], axis=1)

        self.pdtype = make_pdtype(ac_space)
        self.pd = self.pdtype.pdfromflat(pdparam)

        a0 = self.pd.sample()
        neglogp0 = self.pd.neglogp(a0)
        self.initial_state = None

        def step(ob, *_args, **_kwargs):
            a, v, neglogp = sess.run([a0, vf, neglogp0], {X:ob})
            return a, v, self.initial_state, neglogp

        def value(ob, *_args, **_kwargs):
            return sess.run(vf, {X:ob})

        self.X = X
        self.pi = pi
        self.vf = vf
        self.step = step
        self.value = value
