import os
import urllib.request
import subprocess
import glob
import numpy as np


DATA_URI = 'http://redwood.berkeley.edu/cadieu/data/vid075-chunks.tar.gz'
CACHE_PATH = os.path.expanduser('~/data/redwood/')

IM_SZ = 128
PAD_TOP = 14


def _download_tgz_to_npz():
    tmp_tgz = os.path.join(CACHE_PATH, 'tmp.tar.gz')
    urllib.request.urlretrieve(DATA_URI, tmp_tgz)
    os.chdir(CACHE_PATH)
    subprocess.run(['tar', '-zxf', tmp_tgz])
    chunk_fns = glob.glob(os.path.join(CACHE_PATH, 'vid075-chunks', '*'))
    chunk_fns.sort(key=lambda c: int(c.split('/')[-1].lstrip('chunk')))
    # It's big endian, haha... (?is this a matlab thing?)
    chunks = [np.fromfile(f, dtype='>f')
                .astype('=f')
                .reshape(-1, IM_SZ, IM_SZ)
                .transpose(0, 2, 1)[:, PAD_TOP:, :]
              for f in chunk_fns]
    vid = np.concatenate(chunks)
    np.save(os.path.join(CACHE_PATH, 'redwood.npy'), vid)
    os.remove(tmp_tgz)
    for f in chunk_fns:
        os.remove(f)
    os.rmdir(os.path.join(CACHE_PATH, 'vid075-chunks'))


def get_np_dataset():
    if not os.path.exists(os.path.join(CACHE_PATH, 'redwood.npy')):
        _download_tgz_to_npz()

    return np.load(os.path.join(CACHE_PATH, 'redwood.npy'))
