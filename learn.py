import tensorflow as tf
import keras

from DNN.Net import createModel

Model = createModel(input_shape=(256,256,3),n_filters=32,num_layers=5)
Model.summary()
