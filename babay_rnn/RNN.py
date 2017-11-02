"""Recurrent Neural Network (RNN)'s simple implementation

   Author: Yi Zhang <beingzy@gmail.com>
   Date: 2017/11/01
"""
import numpy as np
import scipy as sp


class RNN(object):

    def __init__(self):
        self.h = None
        self.W_xh = None
        self.W_hh = None

    def step(self, x):
        # mathematical operation:
        #      h(t) = tanh(W_hh * h(t-1) + W_xh * x(t))
        # np.tanh() squashs the activations to the range [-1, 1]
        self.h = np.tanh(np.dot(self.W_hh, self.h) + np.dot(self.W_xh, x))
        y = np.dot(self.W_hy, self.h)
        return y
