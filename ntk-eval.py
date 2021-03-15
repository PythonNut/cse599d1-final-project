import neural_tangents as nt
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.stax import logsoftmax

import itertools
import time
from cleverhans.jax.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.jax.attacks.projected_gradient_descent import projected_gradient_descent

from utils import accuracy
from train import validate
import datasets

# Model definition
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Dense(512, 1.0, 0.05),
    nt.stax.Erf(),
    nt.stax.Dense(512, 1.0, 0.05),
    nt.stax.Erf(),
    nt.stax.Dense(10, 1.0, 0.05),
)

apply_fn = jax.jit(apply_fn)
kernel_fn = nt.batch(kernel_fn, 64)

# Generating dataset
x_train, y_train, x_test, y_test = datasets.get_dataset("fashion_mnist", 2048, 128)

# Perturbed datset
x_train_p, y_train_p, x_test_p, y_test_p = datasets.get_dataset(
    "fashion_mnist", 2048, 128, perturb=True
)

# Normal Noise datset
x_train_n, y_train_n, x_test_n, y_test_n = datasets.get_dataset(
    "fashion_mnist", 2048, 128, noise=True
)

# Constructing Kernels
print("=> Computing NTK for train and test")
now = time.time()
g_dd = kernel_fn(x_train, None, "ntk")
g_td = kernel_fn(x_test, x_train, "ntk")
print(f"Took {time.time() - now:0.2f}s")

lam = 1e-4

# Evaluating on test set
predictor = nt.predict.gradient_descent_mse(g_dd, y_train - 0.1, diag_reg=lam)
ntk_out = predictor(None, None, -1, g_td)

print(accuracy(ntk_out, y_test, topk=(1, 5)))


import warnings

warnings.filterwarnings(
    "ignore",
    message="Batch size is reduced from requested 64 to effective 1 to fit the dataset.",
)


def model(x):
    K = kernel_fn(x, x_train, "ntk")
    return predictor(None, None, -1, K)

def do_highfreq_transform(x):
    def do_dct(row):
        return datasets.jdct2(row.reshape(28, 28)).reshape(784)

    # rescale so high-frequencies are easier to change
    gauss_mask = jnp.array(datasets.get_gauss_mask((28, 28)))
    large = 256 * 100
    scale = 1/jnp.maximum(gauss_mask, 1/large).reshape(784)
    scale = jnp.expand_dims(scale, axis=0)
    return jnp.apply_along_axis(do_dct, 1, x) * scale

def undo_highfreq_transform(x):
    def undo_dct(row):
        return datasets.jidct2(row.reshape(28, 28)).reshape(784)

    gauss_mask = jnp.array(datasets.get_gauss_mask((28, 28)))
    large = 256 * 100
    scale = 1/jnp.maximum(gauss_mask, 1/large).reshape(784)
    scale = jnp.expand_dims(scale, axis=0)
    return jnp.apply_along_axis(undo_dct, 1, x/scale)

def model_transformed(u):
    x = undo_highfreq_transform(u)
    K = kernel_fn(x, x_train, "ntk")
    return predictor(None, None, -1, K)

print("=> Running high frequency FGM attack against resulting NTK")
now = time.time()
advantage_ratio = 784/datasets.get_gauss_mask((28, 28)).sum()
x_test_hf_fgm = fast_gradient_method(model_transformed, do_highfreq_transform(x_test), 0.3*advantage_ratio, np.inf)
y_test_hf_fgm = model(x_test_hf_fgm)
print(f"Took {time.time() - now:0.2f}s")
print(accuracy(y_test_hf_fgm, y_test, topk=(1, 5)))

print("=> Running FGM attack against resulting NTK")
now = time.time()
x_test_fgm = fast_gradient_method(model, x_test, 0.3, np.inf)
y_test_fgm = model(x_test_fgm)
print(f"Took {time.time() - now:0.2f}s")
print(accuracy(y_test_fgm, y_test, topk=(1, 5)))

print("=> Running high frequency PGD attack against resulting NTK")
now = time.time()
advantage_ratio = 784/datasets.get_gauss_mask((28, 28)).sum()
x_test_hf_pgd = projected_gradient_descent(model_transformed, do_highfreq_transform(x_test), 0.3*advantage_ratio, 0.01, 40, np.inf)
y_test_hf_pgd = model(x_test_hf_pgd)
print(f"Took {time.time() - now:0.2f}s")
print(accuracy(y_test_hf_pgd, y_test, topk=(1, 5)))

print("=> Running PGD attack against resulting NTK")
now = time.time()
x_test_pgd = projected_gradient_descent(model, x_test, 0.3, 0.01, 40, np.inf)
y_test_pgd = model(x_test_pgd)
print(f"Took {time.time() - now:0.2f}s")
print(accuracy(y_test_pgd, y_test, topk=(1, 5)))

print("=> Computing NTK for high-frq noise test, no noise train")

g_td = kernel_fn(x_test_p, x_train, "ntk")
ntk_out = predictor(None, None, -1, g_td)
print(accuracy(ntk_out, y_test_p, topk=(1, 5)))


print("=> Computing NTK for high-frq noise train and test")
now = time.time()
g_dd_adv = kernel_fn(x_train_p, None, "ntk")
g_td_adv = kernel_fn(x_test_p, x_train_p, "ntk")
print(f"Took {time.time() - now:0.2f}s")

predictor_adv = nt.predict.gradient_descent_mse(g_dd_adv, y_train_p - 0.1, diag_reg=lam)
ntk_out_adv = predictor(None, None, -1, g_td_adv)

print(accuracy(ntk_out_adv, y_test_p, topk=(1, 5)))

print("=> Computing NTK for high-frq noise train and no noise test")
now = time.time()
g_dd_adv = kernel_fn(x_train_p, None, "ntk")
g_td_adv = kernel_fn(x_test, x_train_p, "ntk")
print(f"Took {time.time() - now:0.2f}s")

predictor_adv = nt.predict.gradient_descent_mse(g_dd_adv, y_train_p - 0.1, diag_reg=lam)
ntk_out_adv = predictor(None, None, -1, g_td_adv)

print(accuracy(ntk_out_adv, y_test, topk=(1, 5)))


print("=> Computing NTK for noise test, no noise train")

g_td = kernel_fn(x_test_n, x_train, "ntk")
ntk_out = predictor(None, None, -1, g_td)
print(accuracy(ntk_out, y_test_n, topk=(1, 5)))


print("=> Computing NTK for noise train and test")
now = time.time()
g_dd_adv = kernel_fn(x_train_n, None, "ntk")
g_td_adv = kernel_fn(x_test_n, x_train_n, "ntk")
print(f"Took {time.time() - now:0.2f}s")

predictor_adv = nt.predict.gradient_descent_mse(g_dd_adv, y_train_p - 0.1, diag_reg=1e-3)
ntk_out_adv = predictor(None, None, -1, g_td_adv)

print(accuracy(ntk_out_adv, y_test_n, topk=(1, 5)))

print("=> Computing NTK for noise train and no noise test")
now = time.time()
g_dd_adv = kernel_fn(x_train_n, None, "ntk")
g_td_adv = kernel_fn(x_test, x_train_n, "ntk")
print(f"Took {time.time() - now:0.2f}s")

predictor_adv = nt.predict.gradient_descent_mse(g_dd_adv, y_train_p - 0.1, diag_reg=1e-3)
ntk_out_adv = predictor(None, None, -1, g_td_adv)

print(accuracy(ntk_out_adv, y_test, topk=(1, 5)))
