'''
Created on Mar 25, 2021

@author: MAE82
'''
#*** Mah: set seeds. 
import os
import tensorflow 
from tensorflow.compat.v1 import keras
from tensorflow.compat.v1.keras import layers
from tensorflow.compat.v1.keras import backend as K
import random

# Setting seeds
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"
os.environ['PYTHONHASHSEED'] = str(1234) # Setting seeds
os.environ['tensorflow_DETERMINISTIC_OPS'] = '1' # Setting seeds

random.seed(1234) # Setting seeds
tensorflow.random.set_seed(1234) # Setting seeds
tensorflow.compat.v1.set_random_seed(1234) # Setting seeds



class Linear(keras.layers.Layer):
    
    def __init__(self, embedding_length, name1): 
        super(Linear, self).__init__(name = name1)
        w_init = tensorflow.random_normal_initializer()
        self.w = tensorflow.Variable(
            initial_value=w_init(shape=(embedding_length,), dtype="float32"),
            trainable=True,
        )
        

    def call(self, inputs):
        
        weighted_inputs = [tensorflow.scalar_mul(self.w[i], inputs[:, :, :, i]) for i in range(0, inputs.shape[3])]
        sum_of_embeddings = tensorflow.math.add_n(weighted_inputs)
        sum_of_embeddings = tensorflow.expand_dims(sum_of_embeddings, axis=3)
        sum_of_embeddings = tensorflow.keras.activations.relu(sum_of_embeddings)
        return  sum_of_embeddings




