
from absl import app, flags
import neural_tangents as nt
import jax
import jax.numpy as jnp
import numpy as np
from neural_tangents.utils import kernel
import numpy.random as npr
from jax import jit, grad, random
from jax.experimental import optimizers
from jax.experimental import stax
from jax.experimental.stax import logsoftmax
import matplotlib.pyplot as plt     

from absl import app, flags
import itertools
import torchvision
import torch
import time
import datasets
from cleverhans.jax.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.jax.attacks.projected_gradient_descent import projected_gradient_descent
from keras.datasets import fashion_mnist

from scipy.fftpack import dct, idct

# implement 2D DCT
def dct2(a):
    return dct(dct(a.T, norm='ortho').T, norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho')    

def adding_perturbation_orginal_pixel(img,eta=1.0):
    print(img.shape)
    print(img.shape[0])
    row,col= img.shape[0],img.shape[1]
    shape = row,col
    zeros = np.zeros(shape, dtype=np.int32)
    mean = 0
    var = 1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma,(row,col))
    
    #Apply a filter to high frequency part 
    for i in range(row):
        for j in range(col):
            if (row-i)^2+(row-j)^2 > (row-1)^2:
                gauss[i][j] = 0
   
    for i in range(1,5):
        for j in range(1,5):
            zeros[-i][-j] = gauss[-i][-j] 
    Norm_img = LA.norm(img)

    gauss = gauss.reshape(row,col)
    Norm_Z = LA.norm(gauss)
    noisy = img + eta*(Norm_img)/Norm_Z*gauss
    return noisy
    


def accuracy(logits, one_hot_labels, topk=(1,)):
    preds = jnp.argsort(-logits, axis=1)
    labels = jnp.argmax(one_hot_labels, axis=1).reshape(-1, 1)

    # import pdb; pdb.set_trace()
    accs = []
    for k in topk:
        accs.append(jnp.sum(preds[:, :k] == labels, axis=1).mean())

    return accs



# Model definition
init_fn, apply_fn, kernel_fn = nt.stax.serial(
    nt.stax.Dense(512, 1., 0.05),
    nt.stax.Erf(),
    nt.stax.Dense(512, 1., 0.05),
    nt.stax.Erf(),
    nt.stax.Dense(10, 1., 0.05)
)

apply_fn = jax.jit(apply_fn)
kernel_fn = nt.batch(kernel_fn, 64)

# # Generating dataset
x_train, y_train, x_test, y_test = datasets.get_dataset("fashion_mnist", 1024, 128)

# Constructing Kernels


print("=> Computing NTK for train and test")
now = time.time()
g_dd = kernel_fn(x_train, None, "ntk")
g_td = kernel_fn(x_test, x_train, "ntk")
print(f"Took {time.time() - now:0.2f}s")



# Evaluating on test set
predictor = nt.predict.gradient_descent_mse(g_dd, y_train - 0.1, diag_reg=1e-4)
ntk_out = predictor(None, None, -1, g_td)

print(accuracy(ntk_out, y_test, topk=(1, 5)))


x_train, y_train, x_test, y_test = datasets.get_dataset("fashion_mnist", 1024, 128,perturb=True)

print("=> Computing NTK for high-frq noise train and test")
now = time.time()
g_dd_adv = kernel_fn(x_train, None, "ntk")
g_td_adv = kernel_fn(x_test, x_train, "ntk")
print(f"Took {time.time() - now:0.2f}s")


predictor_adv = nt.predict.gradient_descent_mse(g_dd_adv, y_train - 0.1, diag_reg=1e-4)
ntk_out_adv = predictor(None, None, -1, g_td_adv)

print(accuracy(ntk_out_adv, y_test, topk=(1, 5)))