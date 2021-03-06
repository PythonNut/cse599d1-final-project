from collections import namedtuple
from jax.tree_util import tree_flatten, tree_unflatten


Optimizer = namedtuple("Optimizer", ["step", "internal_state", "lr"])

# Somewhat stateless SGD Step
# TODO: change this to use tree_multimap instead of flatten/unflatten
def sgd_step(
    params,
    gradients,
    lr=0.001,
    weight_decay=1e-4,
    momentum=0.0,
    dampening=0.0,
    internal_state=None,
):
    flat_params, treedef = tree_flatten(params)
    flat_grads, _ = tree_flatten(gradients)

    out_params = []
    internal_state_out = []

    if internal_state is None:
        internal_state = [{} for _ in range(len(flat_params))]

    for w, g, state in zip(flat_params, flat_grads, internal_state):
        internal_state_out.append({})

        delta_w = g

        if weight_decay != 0.0:
            delta_w = delta_w + weight_decay * w

        if momentum != 0.0:
            if "momentum_buffer" not in state:
                internal_state_out[-1]["momentum_buffer"] = delta_w
            else:
                buf = state["momentum_buffer"]
                internal_state_out[-1]["momentum_buffer"] = buf * momentum + delta_w * (
                    1 - dampening
                )

        out_params.append(w - lr * delta_w)

    return tree_unflatten(treedef, out_params), internal_state_out


def SGDOptimizer(
    lr=0.1, weight_decay=0.0, momentum=0.0, dampening=0.0, internal_state=None
):
    def step(params, gradients, lr, internal_state):
        return sgd_step(
            params,
            gradients,
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            dampening=dampening,
            internal_state=internal_state,
        )

    return Optimizer(step=step, lr=lr, internal_state=internal_state)


def update_optimizer_state(optimizer, new_state):
    opt_dict = optimizer._asdict()
    opt_dict['internal_state'] = new_state

    return Optimizer(**opt_dict)