from opt import SGDOptimizer, update_optimizer_state
from utils import accuracy

import neural_tangents as nt
import datasets

from jax.nn import log_softmax

import pickle
import jax.random as random
import jax
import jax.numpy as jnp
import numpy as np
import time
import tqdm


def train(
    train_loader, model, params, criterion, optimizer, epoch, num_images, batch_size
):
    num_batches = (num_images // batch_size) + 1

    # switch to train mode
    def loss_fn(params, x, labels):
        logits = model(params, x)
        loss = criterion(logits, labels)

        return loss, logits

    value_and_grad = jax.jit(jax.value_and_grad(loss_fn, has_aux=True))

    acc1s = []
    acc5s = []
    losses = []

    for i, (images, target) in enumerate(train_loader):
        # compute output
        (loss, output), grad = value_and_grad(params, x=images, labels=target)
        params, opt_state = optimizer.step(
            params, grad, lr=optimizer.lr, internal_state=optimizer.internal_state
        )
        optimizer = update_optimizer_state(optimizer, opt_state)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1s.extend([float(acc1)] * len(images))
        acc5s.extend([float(acc5)] * len(images))
        losses.extend([float(loss)] * len(images))

        tqdm.tqdm.write(
            f"(Train) Epoch {epoch} [{i+1}/{num_batches}] => Loss: {np.mean(losses):0.4f}, Acc@1: {np.mean(acc1s):0.4f}, Acc@5: {np.mean(acc5s):0.4f}",
            end="\r",
        )

    print()

    return params, optimizer, acc1, acc5


def validate(val_loader, model, params, criterion, epoch, batch_size, num_images):
    num_batches = (num_images // batch_size) + 1
    # switch to evaluate mode

    acc1s = []
    acc5s = []
    losses = []
    for i, (images, target) in enumerate(val_loader):
        # compute output
        output = model(params, images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1s.extend([float(acc1)] * len(images))
        acc5s.extend([float(acc5)] * len(images))
        losses.extend([float(loss)] * len(images))

    print(
        f"(Test) Epoch {epoch} [{i+1}/{num_batches}] => Loss: {np.mean(losses):0.4f}, Acc@1: {np.mean(acc1s):0.4f}, Acc@5: {np.mean(acc5s):0.4f}"
    )
    print()

    return np.mean(acc1s), np.mean(acc5s)


def main():
    USE_MSE = False
    EPOCHS = 20

    optimizer = SGDOptimizer(lr=0.1, weight_decay=1e-4)

    if USE_MSE:

        @jax.jit
        def criterion(logits, targets):
            return jnp.mean(jnp.sum((logits - targets) ** 2, axis=1))

    else:

        @jax.jit
        def criterion(logits, targets):
            return -jnp.mean(jnp.sum(log_softmax(logits) * targets, axis=1), axis=0)

    init_fn, apply_fn, _ = nt.stax.serial(
        nt.stax.Dense(512, 1.0, 0.05),
        nt.stax.Erf(),
        nt.stax.Dense(512, 1.0, 0.05),
        nt.stax.Erf(),
        nt.stax.Dense(10, 1.0, 0.05),
    )

    key = random.PRNGKey(0)
    _, params = init_fn(key, (None, 784))

    # Generating dataset
    x_train, y_train, x_test, y_test = datasets.get_dataset("fashion_mnist", 25600, 128)

    for e in range(EPOCHS):
        train_loader = datasets.minibatch(
            x_train, y_train, batch_size=128, train_epochs=1
        )
        val_loader = datasets.minibatch(
            x_test, y_test, batch_size=128, train_epochs=1
        )

        params, optimizer, _, _ = train(
            train_loader,
            apply_fn,
            params,
            criterion,
            optimizer,
            epoch=e,
            num_images=len(x_train),
            batch_size=128,
        )

        validate(
            val_loader,
            apply_fn,
            params,
            criterion,
            epoch=e,
            batch_size=len(x_test),
            num_images=128,
        )

    with open("fashion-mnist-mlp.pkl", "wb+") as f:
        pickle.dump(params, f)


if __name__ == "__main__":
    main()