import itertools
import tensorflow as tf
from dataset import get_np_dataset
from model import first_layer_train_graph, in_n_frames
from util import get_random_patch, patch_whitening_matrix
import matplotlib.pyplot as plt


CROP_SZ = 20


# Data
data = get_np_dataset()
data -= data.mean(axis=0, keepdims=True)

# Learn patchwise whitening matrix
# (whitening_matrix, dewhiten_matrix,
#     zero_phase_matrix) = patch_whitening_matrix(data, in_n_frames, CROP_SZ)
# print(whitening_matrix.shape)

# Build net
net = first_layer_train_graph()

ls = []
with tf.train.MonitoredSession() as sess:
    # Loop forever
    for i in itertools.count():
        patch = get_random_patch(data, in_n_frames, CROP_SZ)
        patch = patch.reshape(in_n_frames, CROP_SZ * CROP_SZ)
        # whitened = whitening_matrix @ patch.T
        whitened = patch.T
        print(whitened.min(), whitened.max())
        _, log_prob_ = sess.run(
            [net.train_op, net.log_prob_opt],
            feed_dict={
                net.input_placeholder: whitened,
            })
        print(f'{i: 7}', log_prob_)
        ls.append(log_prob_)

        if i > 0 and not i % 500:
            A_re_ = sess.run(net.A_re)
            # plt.imshow((dewhiten_matrix @ A_re_[0]).reshape(CROP_SZ, CROP_SZ))
            plt.imshow(A_re_[0].reshape(CROP_SZ, CROP_SZ))
            plt.savefig(f'A{i:07}.png')
            plt.close('all')
            plt.plot(ls[1:])
            plt.savefig('likelihoods.png')
            plt.close('all')
