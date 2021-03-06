import jax.numpy as jnp


def accuracy(logits, one_hot_labels, topk=(1,)):
    preds = jnp.argsort(-logits, axis=1)
    labels = jnp.argmax(one_hot_labels, axis=1).reshape(-1, 1)

    # import pdb; pdb.set_trace()
    accs = []
    for k in topk:
        accs.append(jnp.sum(preds[:, :k] == labels, axis=1).mean())

    return accs
