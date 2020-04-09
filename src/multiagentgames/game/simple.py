from jax import jit, grad, vmap, random, jacrev, jacobian, jacfwd
from functools import partial
import jax
import jax.numpy as jp
import jax.scipy as jsp
from jax.experimental import stax # neural network library
from jax.experimental.stax import GeneralConv, Conv, ConvTranspose, Dense, MaxPool, Relu, Flatten, LogSoftmax, LeakyRelu, Dropout, Tanh, Sigmoid, BatchNorm # neural network layers
from jax.nn import softmax, sigmoid
from jax.nn.initializers import zeros
import matplotlib.pyplot as plt # visualization
import numpy as np
from jax.experimental import optimizers
from jax.tree_util import tree_multimap  # Element-wise manipulation of collections of numpy arrays
from jax.ops import index, index_add, index_update
import os, time

def imp():
  dims = [1, 1]
  payout_mat_1 = jp.array([[1,-1],[-1,1]])
  payout_mat_2 = -payout_mat_1
  def Ls(th):
    # th=jp.stack(th).reshape(-1)
    p_1, p_2 = sigmoid(th[0]), sigmoid(th[1])
    x, y = jp.array([p_1, 1-p_1]), jp.array([p_2, 1-p_2])
    # print(x.shape,y.shape,payout_mat_1.shape,payout_mat_2.shape)
    L_1 = jp.dot(jp.dot(x.T, payout_mat_1), y)
    L_2 = jp.dot(jp.dot(x.T, payout_mat_2), y)
    return jp.array([L_1.reshape(-1)[0], L_2.reshape(-1)[0]])
  return dims, Ls

def ipd(gamma=0.96):
  dims = [5, 5]
  payout_mat_1 = jp.array([[-1,-3],[0,-2]])
  payout_mat_2 = payout_mat_1.T
  def Ls(th):
    p_1_0 = sigmoid(th[0][0:1])
    p_2_0 = sigmoid(th[1][0:1])
    p = jp.stack([p_1_0*p_2_0, p_1_0*(1-p_2_0), (1-p_1_0)*p_2_0, (1-p_1_0)*(1-p_2_0)], axis=1)
    # print('p',p,p.shape)
    p_1 = jp.reshape(sigmoid(th[0][1:5]), (4, 1))
    p_2 = jp.reshape(sigmoid(th[1][1:5]), (4, 1))
    P = jp.stack([p_1*p_2, p_1*(1-p_2), (1-p_1)*p_2, (1-p_1)*(1-p_2)], axis=1).reshape((4,4))
    # print('P',P,P.shape)
    # print('inv', jsp.linalg.inv(jp.eye(4)-gamma*P), jsp.linalg.inv(jp.eye(4)-gamma*P).shape)
    M = -jp.dot(p, jsp.linalg.inv(jp.eye(4)-gamma*P))
    # print('M',M)
    L_1 = jp.dot(M, jp.reshape(payout_mat_1, (4, 1)))
    L_2 = jp.dot(M, jp.reshape(payout_mat_2, (4, 1)))
    # print('L_1',L_1.reshape(-1)[0])
    # print('L_2',L_2.reshape(-1)[0])
    return jp.array([L_1.reshape(-1)[0], L_2.reshape(-1)[0]])
  return dims, Ls

def tandem():
  dims = [1, 1]
  def Ls(th):
    x, y = th[0], th[1]
    L_1 = (x+y)**2-2*x
    L_2 = (x+y)**2-2*y
    return jp.array([L_1.reshape(-1)[0], L_2.reshape(-1)[0]])
  return dims, Ls