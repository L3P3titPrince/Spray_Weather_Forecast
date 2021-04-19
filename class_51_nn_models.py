from class_31_hyperparameters import HyperParamters

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

class NNModels(HyperParamters):
    """

    """
    def __init__(self):
        """

        """
        HyperParamters.__init__(self)


    def mlp_model(self):
        """
        Multi-Layer Perceptrons is the simple 

        :return:
        """
        model = None
        input_layer_1 = layers.Input(shape=(26,), name='input_layer_1')
        dense_layer_2 = layers.Dense(units=128, activation='relu', name='dense_layer_2')(input_layer_1)
        dense_layer_3 = layers.Dense(units=32, activation='relu', name='dense_layer_3')(dense_layer_2)
        output_layer_4 = layers.Dense(units=self.NN_OUTPUT, activation='softmax', name='output_layer_4')(dense_layer_3)

        model = keras.Model(inputs=input_layer_1, outputs=output_layer_4, name='nn_model')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2, decay=1e-5),
                      loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
        model.summary()

        return model


