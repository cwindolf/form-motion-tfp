import numpy as np
import scipy.linalg as la


def get_random_patch(data, n_frames, crop_sz, margin=4):
    t, n, m = data.shape
    start_frame = np.random.randint(t - n_frames)
    start_i = margin + np.random.randint(n - 2 * margin - crop_sz)
    start_j = margin + np.random.randint(m - 2 * margin - crop_sz)
    return data[start_frame:start_frame + n_frames,
                start_i:start_i + crop_sz,
                start_j:start_j + crop_sz]


def patch_whitening_matrix(data,
                           n_frames,
                           crop_sz,
                           pixel_noise_var_fraction=0.01,
                           pixel_noise_var_cutoff_ratio=1.25,
                           X_noise_fraction=8.0,
                           X_noise_var=0.01,
                           num_patches=200000):
    '''
    Let `data` already be centered and scaled.
    Then, don't need to store all the patches in memory like they do...
    This does some whitening and low pass filter.
    '''
    C = np.zeros((crop_sz * crop_sz, crop_sz * crop_sz))
    for i in range(num_patches):
        patch = get_random_patch(data, n_frames, crop_sz)
        patch = patch.reshape(n_frames, crop_sz * crop_sz)
        C += patch.T @ patch

    pixel_variance = np.diag(C).mean()
    pixel_noise_variance = pixel_noise_var_fraction * pixel_variance

    d, E = la.eigh(C)
    d = d[::-1]
    E = E[:, ::-1]

    variance_cutoff = pixel_noise_var_cutoff_ratio * pixel_noise_variance
    M = (d > variance_cutoff).sum()
    print('Whitening M:', M)

    var_X = d[0:M]
    factors = np.sqrt(var_X).real
    E = E[:, 0:M]
    D = np.diag(factors)

    whitening_matrix = D @ E.T
    dewhiten_matrix = E @ la.inv(D)
    zero_phase_matrix = E @ D @ E.T

    return whitening_matrix, dewhiten_matrix, zero_phase_matrix
