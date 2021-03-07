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

import pathlib
import pickle
assert pathlib.Path("fashion-mnist-mlp.pkl").exists()


init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Dense(512, 1.0, 0.05),
    nt.stax.Erf(),
    nt.stax.Dense(512, 1.0, 0.05),
    nt.stax.Erf(),
    nt.stax.Dense(10, 1.0, 0.05),
)


with open("fashion-mnist-mlp.pkl", "rb") as f:
    params = pickle.load(f)


@jax.jit
def criterion(logits, targets):
    return jnp.mean(jnp.sum((logits - targets) ** 2, axis=1))


def model(x):
    return apply_fn(params, x)

print("Uncorrupted Test Accuracy")
print("="*80)
x_train, y_train, x_test, y_test = datasets.get_dataset("fashion_mnist", 1024, 128)

validate(
    val_loader=datasets.minibatch(
        x_test, y_test, batch_size=128, train_epochs=1, key=None
    ),
    model=apply_fn,
    params=params,
    criterion=criterion,
    epoch=20,
    batch_size=128,
    num_images=len(x_test),
)

print("Corrupted Test Accuracy")
print("="*80)
x_train, y_train, x_test, y_test = datasets.get_dataset("fashion_mnist", 1024, 128, perturb=True)

validate(
    val_loader=datasets.minibatch(
        x_test, y_test, batch_size=128, train_epochs=1, key=None
    ),
    model=apply_fn,
    params=params,
    criterion=criterion,
    epoch=20,
    batch_size=128,
    num_images=len(x_test),
)

x_train, y_train, x_test, y_test = datasets.get_dataset("fashion_mnist", 1024, 128)

print("=> Running FGM attack against resulting NTK")
now = time.time()
x_test_fgm = fast_gradient_method(model, x_test, 0.3, np.inf)
y_test_fgm = model(x_test_fgm)
print(f"Took {time.time() - now:0.2f}s")
print(accuracy(y_test_fgm, y_test, topk=(1, 5)))


print("=> Running PGD attack against resulting NTK")
now = time.time()
x_test_pgd = projected_gradient_descent(model, x_test, 0.3, 0.01, 40, np.inf)
y_test_pgd = model(x_test_pgd)
print(f"Took {time.time() - now:0.2f}s")
print(accuracy(y_test_pgd, y_test, topk=(1, 5)))