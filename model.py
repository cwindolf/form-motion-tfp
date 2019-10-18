import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from collections import namedtuple

tfd = tfp.distributions
tfb = tfp.bijectors


# ----------------------- model config params -------------------------

# Input movie shape
in_n_frames = 128
in_n_features = 400

# First layer
n_basis_A = 1024
a_lambda_ = 10.0
a_cauchy_sigma_ = 0.4
a_beta_ = 0.5
var_N_ = 0.2
A_eta_ = 0.05

# Second layer
n_basis_B = 625
n_basis_D = 625

# TF-ify
var_N = tf.constant(var_N_, name='var_N')
a_cauchy_sigma = tf.constant(a_cauchy_sigma_, name='a_cauchy_sigma')
a_lambda = tf.constant(a_lambda_, name='a_lambda')
a_beta = tf.constant(a_beta_, name='a_beta')


# --------------------------- data holders ----------------------------

FirstLayer = namedtuple(
    'FirstLayer',
    ['A_re', 'A_im', 'input_placeholder', 'map_a', 'map_phi',
     'train_op', 'log_prob_opt'])


# ------------------------ sparsity / slowness ------------------------


def log_cauchy(x, sigma):
    return tf.log1p(tf.square(x / sigma))


def Sp_a(a):
    '''Sparse prior on a'''
    return a_lambda * tf.reduce_sum(log_cauchy(a, a_cauchy_sigma))


def Sl_a(a):
    '''Slow prior on a'''
    return a_beta * tf.reduce_sum(
        tf.squared_difference(
            a[:, 1:],
            a[:, :-1]))


# ---------------------- layer energy functions -----------------------


def first_layer_log_prob(I, a, phi, A_re, A_im):
    '''First layer's log probability (from 3.7)

    Arguments
    ---------
    I : tensor (D, T)
        Input movie
    a, phi : tensor (K, T)
        Amplitudes and phases
    A_re, A_im : tensor (K, D)
        Basis functions
    '''
    a_cos_phi = tf.expand_dims(a * tf.cos(phi), axis=-1)
    a_sin_phi = tf.expand_dims(a * tf.sin(phi), axis=-1)
    print('acp', a_cos_phi.shape, a_sin_phi.shape)
    print('Are', A_re.shape, A_im.shape)
    reco_re = a_cos_phi * tf.expand_dims(A_re, axis=1)
    reco_im = a_sin_phi * tf.expand_dims(A_im, axis=1)
    print('reco_re', reco_re.shape, reco_im.shape)
    reco = tf.reduce_sum(reco_re + reco_im, axis=0)
    print('reco', reco.shape)
    print('input', I.shape)

    reco_log_prob = (
        (1.0 / var_N)
        * tf.reduce_sum(
            tf.squared_difference(
                I,
                tf.transpose(reco, (1, 0)))))

    sparsity_log_prob = Sp_a(a)

    slowness_log_prob = Sl_a(a)

    return reco_log_prob + sparsity_log_prob + slowness_log_prob


# ------------------------- training graphs ---------------------------


def first_layer_train_graph():
    '''Build ops for running a training step on the first layer'''
    with tf.name_scope('A'):
        input_placeholder = tf.placeholder(
            tf.float32,
            shape=(in_n_features, in_n_frames),
            name='input_placeholder')

        A_re_init = np.random.uniform(size=(n_basis_A, in_n_features))
        A_re_init /= np.square(A_re_init).sum()
        A_im_init = np.random.uniform(size=(n_basis_A, in_n_features))
        A_im_init /= np.square(A_im_init).sum()

        A_re = tf.get_variable(
            'A_re',
            initializer=A_re_init.astype(np.float32))
        A_im = tf.get_variable(
            'A_im',
            initializer=A_im_init.astype(np.float32))

        def objective_and_grads(z, ret_grads=True):
            z_ = tf.reshape(z, (2 * n_basis_A, in_n_frames))
            a = z_[:n_basis_A]
            phi = z_[n_basis_A:]
            obj = first_layer_log_prob(input_placeholder, a, phi, A_re, A_im)
            if not ret_grads:
                return obj
            grads = tf.gradients(obj, z)[0]
            return obj, grads

        def objective(z):
            return objective_and_grads(z, ret_grads=False)

        chain_init_a = tf.random.uniform((n_basis_A, in_n_frames))
        chain_init_a = (chain_init_a
                        / tf.reduce_sum(tf.abs(chain_init_a),
                                        axis=1, keepdims=True))
        chain_init_phi = tf.random.uniform(
            (n_basis_A, in_n_frames),
            maxval=(2 * np.pi))
        chain_init = tf.concat(
            [chain_init_a, chain_init_phi],
            axis=0)
        chain_init = tf.reshape(chain_init, (2 * n_basis_A * in_n_frames,))

        # Get a sample
        # BFGS style
        # results = tfp.optimizer.lbfgs_minimize(
        #     objective_and_grads,
        #     initial_position=chain_init,
        #     name='lbfgs_sampler')
        # res_z = results.position

        # MCMC style
        hmc = tfp.mcmc.HamiltonianMonteCarlo(
            target_log_prob_fn=objective,
            step_size=tf.Variable(1.0),
            num_leapfrog_steps=2,
            step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy(
                2))
        # hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        #     inner_kernel=hmc,
        #     num_adaptation_steps=100)
        results = tfp.mcmc.sample_chain(
            1,
            chain_init,
            kernel=hmc,
            num_burnin_steps=10)
        res_z = results[0]

        # Map z back to a,phi.
        # Need to make sure Adam doesn't try to backprop through
        # the sampling step, so use stop_gradient.
        map_z = tf.reshape(res_z, (2 * n_basis_A, in_n_frames))
        map_a = tf.stop_gradient(map_z[:n_basis_A])
        map_phi = tf.stop_gradient(map_z[n_basis_A:])

        # Update weights. Again, not using the grads they give, just letting
        # tf do the work.
        log_prob_opt = first_layer_log_prob(
            input_placeholder, map_a, map_phi, A_re, A_im)
        train_op = tf.train.AdamOptimizer(0.5).minimize(
            -log_prob_opt,
            var_list=[A_re, A_im],
            name='train_op')
        # Sphere As and swap re/im parts like matlab code does
        train_and_project_op = tf.group([
            train_op,
            A_re.assign(tf.math.l2_normalize(A_im, axis=1)),
            A_im.assign(tf.math.l2_normalize(A_re, axis=1)),
        ])

    return FirstLayer(
        A_re=A_re, A_im=A_im, map_a=map_a, map_phi=map_phi,
        input_placeholder=input_placeholder, train_op=train_and_project_op,
        log_prob_opt=log_prob_opt)
