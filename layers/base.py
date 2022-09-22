import tensorflow as tf
from tensorflow import keras


def get_FeedForwardNN(layer_input, units,
                      activation="relu", depth=1,
                      no_output_activation=False, kernel_initializer="glorot_uniform",
                      name='anonymous'):
    if tf.rank(units) == 0:
        units = [units] * depth

    output = layer_input
    for i in range(len(units)):
        if i == len(units) - 1 and no_output_activation:
            output = keras.layers.Dense(
                units[i], name=f"{name}/dense_{i+1}", kernel_initializer=kernel_initializer
            )(output)
        else:
            output = keras.layers.Dense(
                units[i], name=f"{name}/dense_{i+1}", activation=activation, kernel_initializer=kernel_initializer
            )(output)
    return output


class FeedForwardNN(keras.layers.Layer):
    def __init__(self, units,
                 activation="relu", depth=1,
                 no_output_activation=False, kernel_initializer='glorot_uniform', **kwargs):
        super(FeedForwardNN, self).__init__(**kwargs)
        self.supports_masking = True

        if tf.rank(units) == 0:
            units = [units] * depth

        self._denses = []
        for i in range(len(units)):
            if i == len(units) - 1 and no_output_activation:
                self._denses.append(
                    keras.layers.Dense(units[i], name=f"dense_{i+1}", kernel_initializer=kernel_initializer)
                )
            else:
                self._denses.append(
                    keras.layers.Dense(
                        units[i], name=f"dense_{i+1}", activation=activation, kernel_initializer=kernel_initializer
                    )
                )

    def call(self, inputs, mask=None):
        output = inputs
        for dense in self._denses:
            output = dense(output)
        return output


class FieldEmbedding(keras.layers.Layer):
    def __init__(self, embed_dim, input_dims, **kwargs):
        self.embed_dim = embed_dim
        self.input_dims = input_dims
        super(FieldEmbedding, self).__init__(**kwargs)

        self.n_fields = len(input_dims)
        self.embedding_layers = []
        for i in range(self.n_fields):
            self.embedding_layers.append(keras.layers.Embedding(input_dim=input_dims[i], output_dim=embed_dim))

    def call(self, inputs, **kwargs):
        return tf.stack([self.embedding_layers[i](inputs[:, i]) for i in range(self.n_fields)], axis=1)

    def get_config(self):
        config = super(FieldEmbedding, self).get_config()
        config.update({'embed_dim': self.embed_dim, "input_dims": self.input_dims})
        return config
