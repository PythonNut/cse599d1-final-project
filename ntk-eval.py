import neural_tangents as nt
import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.stax import logsoftmax

import itertools
import time
from cleverhans.jax.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.jax.attacks.projected_gradient_descent import projected_gradient_descent

from utils import *
from train import validate
import datasets

from matplotlib import pyplot as plt

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


def kernel_fit(x_tr, y_tr, lam=1e-3):
    g_dd = kernel_fn(x_tr, None, "ntk")
    predictor = nt.predict.gradient_descent_mse(g_dd, y_tr - 0.1, diag_reg=lam)

    def model(x_te):
        g_td = kernel_fn(x_te, x_tr, "ntk")
        return predictor(None, None, -1, g_td)

    return model


n_train, n_test = 2048, 128
# Generating dataset
x_train, y_train, x_test, y_test = datasets.get_dataset(
    "fashion_mnist", n_train, n_test
)

# Perturbed datset
x_train_p, y_train_p, x_test_p, y_test_p = datasets.get_dataset(
    "fashion_mnist", n_train, n_test, perturb=True
)

# Normal noise datset
x_train_n, y_train_n, x_test_n, y_test_n = datasets.get_dataset(
    "fashion_mnist", n_train, n_test, noise=True
)

# Constructing Kernels
print("=> Computing NTK for train and test")
with print_time():
    model = kernel_fit(x_train, y_train)

# Evaluating on test set
print(accuracy(model(x_test), y_test, topk=(1, 5)))

print("=> Running high frequency FGM attack against resulting NTK")
with print_time():
    x_test_hf_fgm = fast_gradient_method(
        model_highfreq_transformed(model),
        do_highfreq_transform(x_test),
        0.3,
        np.inf,
    )
print(accuracy(model(x_test_hf_fgm), y_test, topk=(1, 5)))

print("=> Running FGM attack against resulting NTK")
with print_time():
    x_test_fgm = fast_gradient_method(model, x_test, 0.3, np.inf)
print(accuracy(model(x_test_fgm), y_test, topk=(1, 5)))

print("=> Running high frequency PGD attack against resulting NTK")
with print_time():
    x_test_hf_pgd = projected_gradient_descent(
        model_highfreq_transformed(model),
        do_highfreq_transform(x_test),
        0.3,
        0.01,
        40,
        np.inf,
    )
print(accuracy(model(x_test_hf_pgd), y_test, topk=(1, 5)))

print("=> Running PGD attack against resulting NTK")
with print_time():
    x_test_pgd = projected_gradient_descent(model, x_test, 0.3, 0.01, 40, np.inf)
print(accuracy(model(x_test_pgd), y_test, topk=(1, 5)))

train_sets, test_sets = [
    (x_train, y_train, "clean"),
    (x_train_p, y_train_p, "normal"),
    (x_train_n, y_train_n, "high freq"),
], [
    (x_test, y_test, "clean"),
    (x_test_p, y_test_p, "normal"),
    (x_test_n, y_test_n, "high freq"),
]

for (x_tr, y_tr, name_tr), (x_te, y_te, name_te) in itertools.product(
    train_sets, test_sets
):
    print(f"=> Running {name_tr} train, {name_te} test")
    print(accuracy(kernel_fit(x_tr, y_tr)(x_te), y_te, topk=(1, 5)))
