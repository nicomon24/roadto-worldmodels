
# coding: utf-8

# # LSTM XOR
# We want to implement an LSTM-RNN in TF to predict the XOR operation over a sequence of fixed length (for now) of bits.
# First we need to generate these sequences and to compute the XOR itself.
# 
# Remarks:
# - The XOR operation has a non-linear decision hyperplane

# In[1]:


import tensorflow as tf
import numpy as np

