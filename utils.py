import timeit
import itertools

import jax
import numpy as np
import jax.numpy as jnp

from scipy.fftpack import dct, idct
from contextlib import contextmanager

import warnings

warnings.filterwarnings(
    "ignore",
    message="Batch size is reduced from requested 64 to effective 1 to fit the dataset.",
)

def get_gauss_mask(shape, eta=0.5):
    row, col = shape
    gauss = np.ones((row, col))

    # Apply a filter to high frequency part
    for i in range(row):
        for j in range(col):
            if (row - i) ** 2 + (row - j) ** 2 > (row - 1) ** 2:
                gauss[i][j] = 0

    return gauss

def do_highfreq_transform(x):
    def do_dct(row):
        return jdct2(row.reshape(28, 28)).reshape(784)

    # rescale so high-frequencies are easier to change
    gauss_mask = jnp.array(get_gauss_mask((28, 28)))
    large = 256 * 100
    scale = 1/jnp.maximum(gauss_mask, 1/large).reshape(784)
    scale = jnp.expand_dims(scale, axis=0)
    return jnp.apply_along_axis(do_dct, 1, x) * scale

def undo_highfreq_transform(x):
    def undo_dct(row):
        return jidct2(row.reshape(28, 28)).reshape(784)

    gauss_mask = jnp.array(get_gauss_mask((28, 28)))
    large = 256 * 100
    scale = 1/jnp.maximum(gauss_mask, 1/large).reshape(784)
    scale = jnp.expand_dims(scale, axis=0)
    return jnp.apply_along_axis(undo_dct, 1, x/scale)

def model_highfreq_transformed(model):
    return lambda u: model(undo_highfreq_transform(u))

def jdct(x, norm=None):
    assert len(x.shape) == 2
    n, d = x.shape
    u = jnp.zeros((n, 4*d))
    u = jax.ops.index_update(u, jax.ops.index[:, 1:2*d:2], x)
    u = jax.ops.index_update(u, jax.ops.index[:, 2*d+1::2], x[:, ::-1])
    U = jnp.fft.rfft(u)[:, :d]

    if norm == "ortho":
        scale = jnp.ones(d) / jnp.sqrt(2*d)
        scale = jax.ops.index_mul(scale, jax.ops.index[0], 1/jnp.sqrt(2))
        return U.real * jnp.expand_dims(scale, axis=0)

    return U.real

def jidct(U, norm=None):
    assert len(U.shape) == 2
    n, d = U.shape

    if norm == "ortho":
        scale = jnp.ones(d) / jnp.sqrt(2*d)
        scale = jax.ops.index_mul(scale, jax.ops.index[0], 1/jnp.sqrt(2))
        U /= jnp.expand_dims(scale, axis=0)

    U = jnp.hstack([U, jnp.zeros((n, 1)), -U[:, ::-1]])
    u = jnp.fft.irfft(U)[:, 1:2*d:2]
    return u

def jdct2(a):
    return jdct(jdct(a.T, norm="ortho").T, norm="ortho")

def jidct2(a):
    return jidct(jidct(a.T, norm="ortho").T, norm="ortho")

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm="ortho").T, norm="ortho")


# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm="ortho").T, norm="ortho")


def accuracy(logits, one_hot_labels, topk=(1,)):
    preds = jnp.argsort(-logits, axis=1)
    labels = jnp.argmax(one_hot_labels, axis=1).reshape(-1, 1)

    # import pdb; pdb.set_trace()
    accs = []
    for k in topk:
        accs.append(jnp.sum(preds[:, :k] == labels, axis=1).mean())

    return accs

@contextmanager
def print_time(fmt="Took {:.2f}s"):
    start_time = timeit.default_timer()
    try:
        yield
    finally:
        end_time = timeit.default_timer()
        print(fmt.format(end_time - start_time))
