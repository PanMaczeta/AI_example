import tensorflow as tf
import keras
from keras import layers

def createModel(input_shape     : tuple,
                #output_shape    : tuple,
                config_available: bool       = False,
                model_name      : str        = 'default',
                n_filters       : int        = 32,
                num_layers      : int        = 3,
                max_pool_list   : list[bool] = None,
                debug           : bool       = False) -> keras.Sequential:
    
    if debug: print(f'Initial variable list:\n Input shape:{input_shape} \n Config availability: {config_available}\n Model name: {model_name}\n Decreasing N Filters number:{n_filters}\n MaxPool List: {max_pool_list}')
    if max_pool_list == None:
        max_pool_list = [True for x in range(0,num_layers)]
        if debug: print('MaxPool on every Convolutional layer added...')
    if not len(max_pool_list) == num_layers:
        raise ValueError(f'Number of layers:{num_layers} , MaxPool list shoud be same size, {len(max_pool_list)} was given')
    '''
        input_shape     : Tuple of input shape
        model_name      : Name your model. Default = 'pose2D'
        n_filters       : Number of initial nFilters number. Default = 32
        num_layers      : Number of Convolution layers. Default = 3
                        '''
    Model = keras.Sequential(name = model_name)

    Model.add(keras.Input(shape=input_shape))

    for i in range(0,num_layers):
        if i == 0:
          Model.add(layers.Conv2D(filters= n_filters,activation = 'linear',kernel_size=(3,3)))
        else:
          Model.add(layers.Conv2D(filters= n_filters,activation = 'linear',kernel_size=(3,3)))
        Model.add(layers.BatchNormalization())
        Model.add(layers.Activation(activation='relu'))
        if max_pool_list[i]:
            Model.add(layers.MaxPool2D(pool_size=(2,2)))
    Model.add(layers.Flatten())
    Model.add(layers.Dense(32, activation='relu'))
    Model.add(layers.Dropout(rate= 0.5))
    Model.add(layers.Dense(64,activation='relu'))
    Model.add(layers.Reshape(target_shape=(8,8)))
    return Model