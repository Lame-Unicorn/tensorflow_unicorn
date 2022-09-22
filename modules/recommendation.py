import tensorflow as tf
from tensorflow import keras


class FMCrossing(keras.layers.Layer):
    """Factorization Machines - Rendle 2010.
    See https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf or
    [因子分解机（Factorization Machines）]
    (http://www.lameunicorn.cn/2021/09/23/%E5%9B%A0%E5%AD%90%E5%88%86%E8%A7%A3%E6%9C%BA%EF%BC%88Factorization-Machines%EF%BC%89/)
    for more details about FM.

    FMCrossing take field embeddings as input and produce field-level feature crossing.
    Assume input shape is (None, n_fields, embed_dim), output shape will be (None, n_fields * (n_fields + 1)/ 2.
    """
    def __init__(self, **kwargs):
        super(FMCrossing, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.input_size = input_shape[1]
        self.output_size = (self.input_size + 1) * self.input_size // 2

    def call(self, x):
        # Shape of x: (batch_size, input_size, embed_dim)

        x = tf.linalg.matmul(x, x, transpose_b=True)
        # Shape of x: (batch_size, input_size, input_size)

        x = keras.layers.Flatten()(
            tf.reverse(x, axis=[1])
        )
        # Shape of x: (batch_size, input_size**2)

        # Shape of x: (batch_size, input_size * (input_size + 1)/ 2)
        x = x[:, :self.output_size]
        return x


class DCNCrossing(keras.layers.Layer):
    """Deep & Cross Network for Ad Click Predictions - Wang et al. 2017.
    See [arXiv:1708.05123](https://arxiv.org/abs/1708.05123) or
    [Wide&Deep联合训练模型](http://www.lameunicorn.cn/2021/10/27/Wide-Deep%E8%81%94%E5%90%88%E8%AE%AD%E7%BB%83%E6%A8%A1%E5%9E%8B/)
    for more details about DCN.

    DCNCrossing layer implement the mathematical operations:
    $$x_l=x_0\odot(W_lx_{l-1}+b)+x_{l-1}$$
    or this operation:
    `xl = x0 * Dense(xlm1) + xlm1`

    Input of the whole DCNCrossing part is the concatenated field embeddings (instead of field embeddings) and DCNCrossing produce one step higher
    feature interactions.
    """

    def __init__(self, **kwargs):
        super(DCNCrossing, self).__init__(**kwargs)

    def build(self, input_shape):
        self.dense = keras.layers.Dense(input_shape[0][-1], activation=None)

    def call(self, inputs):
        return self.dense(inputs[0]) * inputs[1] + inputs[0]


class AUGRUCell(keras.layers.Layer):
    """Deep Interest Evolution Network for Click-Through Rate Prediction - Zhou et al. 2019.
    See https://doi.org/10.1609/aaai.v33i01.33015941 or
    [Attention与推荐系统][http://www.lameunicorn.cn/2021/11/04/Attention%E4%B8%8E%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F/]
    for more details about DIEN.

    AUGRUCell take attention weights as extra input and scale update gate.
    """
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(AUGRUCell, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]

        self.kernel = self.add_weight(
            shape=(input_dim, self.units * 3),
            name='kernel',
            initializer='glorot_uniform',
        )

        self.input_bias = self.add_weight(
            shape=(self.units * 3,),
            name='input_bias',
            initializer='zeros',
        )

        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units * 3),
            name='recurrent_kernel',
            initializer='orthogonal',
        )

        self.recurrent_bias = self.add_weight(
            shape=(self.units * 2,),
            name='recurrent_bias',
            initializer='zeros',
        )

    def call(self, inputs, states, training=None):
        h_tm1 = states[0] if tf.nest.is_nested(states) else states

        matrix_x = tf.tensordot(inputs[0], self.kernel, axes=[[1], [0]])
        matrix_x = tf.add(matrix_x, self.input_bias)

        x_z, x_r, x_h = tf.split(matrix_x, 3, axis=-1)

        matrix_inner = tf.tensordot(h_tm1, self.recurrent_kernel[:, :self.units * 2], axes=1)
        matrix_inner = tf.add(matrix_inner, self.recurrent_bias)

        recurrent_z, recurrent_r = tf.split(
            matrix_inner, 2, axis=-1
        )

        z = keras.activations.hard_sigmoid(x_z + recurrent_z) * inputs[1]
        r = keras.activations.hard_sigmoid(x_r + recurrent_r)

        recurrent_h = tf.tensordot(
            r * h_tm1, self.recurrent_kernel[:, 2 * self.units:], axes=1
        )

        hh = keras.activations.tanh(x_h + recurrent_h)

        h = z * h_tm1 + (1 - z) * hh
        new_state = [h] if tf.nest.is_nested(states) else h
        return h, new_state

    def get_config(self):
        config = super(AUGRUCell, self).get_config()
        config.update({'units': self.units})
        return config


class CINLayer(keras.layers.Layer):
    """xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems - Lian et al. 2018.
    See https://arxiv.org/abs/1803.05170 for more details about Compressed Interaction Network (CIN) and xDeepFM.

    CIN is trying to provide explicit high-order interactions in a more effective way of DCN. CIN takes field embeddings
    as input and return 

    """
    def __init__(self, compress_dim, **kwargs):
        self.compress_dim = compress_dim
        super(CINLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(self.compress_dim, input_shape[0][1], input_shape[1][1]),
            name='kernel',
            initializer='glorot_uniform',
        )

    def call(self, inputs):
        outer_prod = tf.expand_dims(inputs[0], 2) * tf.expand_dims(inputs[1], 1)
        output_mat = tf.tensordot(outer_prod, self.kernel, axes=[[1, 2], [1, 2]])
        output_mat = tf.transpose(output_mat, [0, 2, 1])
        final_output = tf.reduce_sum(output_mat, axis=-1)
        return output_mat, final_output

    def get_config(self):
        config = super(CINLayer, self).get_config()
        config.update({"compress_dim": self.compress_dim})
        return config


class CIN(keras.layers.Layer):
    def __init__(self, compress_dims, **kwargs):
        self.compress_dims = compress_dims
        super(CIN, self).__init__(**kwargs)

        self.cin_layers = []
        for i, dim in enumerate(self.compress_dims):
            self.cin_layers.append(CINLayer(dim, name=f"{self.name}-{i + 1}"))

    def build(self, input_shape):
        for i, layer in enumerate(self.cin_layers):
            layer.build((
                (input_shape[0], self.compress_dims[i - 1] if i > 0 else input_shape[1], input_shape[2]),
                input_shape
            ))

    def call(self, inputs):
        outputs = []
        x = inputs
        for layer in self.cin_layers:
            x, output = layer((x, inputs))
            outputs.append(output)
        return tf.concat(outputs, axis=1)

    def get_config(self):
        config = super(CIN, self).get_config()
        config.update({"compress_dims": self.compress_dims})
        return config


class AttentionScore(keras.layers.Layer):
    """Deep Interest Evolution Network for Click-Through Rate Prediction - Zhou et al. 2019.
    See https://doi.org/10.1609/aaai.v33i01.33015941 or
    [Attention与推荐系统][http://www.lameunicorn.cn/2021/11/04/Attention%E4%B8%8E%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F/]
    for more details about DIEN.

    AttentionScore takes an ad embedding and an history behavior embedding sequence as inputs and returns attention
    weights using ad embedding as query and sequence as keys.
    AttentionScore allows ad embeddings and embedding sequence have different embedding dimension by applying linear
    transformation.
    """
    def __init__(self, value_dim, **kwargs):
        super(AttentionScore, self).__init__(**kwargs)
        self.value_dim = value_dim
        self.kernel = keras.layers.Dense(value_dim, activation=None, use_bias=False)

    def call(self, inputs):
        attention_logits = keras.layers.dot(
            [self.kernel(inputs[0]), inputs[1]], axes=[1, 2]
        )
        return tf.expand_dims(tf.nn.softmax(attention_logits, axis=-1), axis=-1)

    def get_config(self):
        config = super(AttentionScore, self).get_config()
        config.update({'value_dim': self.value_dim})
        return config


class AUGRU(keras.layers.Layer):
    """Deep Interest Evolution Network for Click-Through Rate Prediction - Zhou et al. 2019.
    See https://doi.org/10.1609/aaai.v33i01.33015941 or
    [Attention与推荐系统][http://www.lameunicorn.cn/2021/11/04/Attention%E4%B8%8E%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F/]
    for more details about DIEN.

    AUGRU takes historical interests embedding sequence (history behaviour embedding sequence after GRU layer) as input
    and return the deep interest evolving embedding sequence.
    """
    def __init__(self, units, return_attention_weights=False, return_sequences=False, return_state=False,
                 stateful=False, **kwargs):
        self.units = units
        self.return_attention_weights = return_attention_weights
        self.return_sequences = return_sequences
        self.return_state = return_state
        self.stateful = stateful
        super(AUGRU, self).__init__(**kwargs)

        self.rnn = keras.layers.RNN(
            AUGRUCell(self.units),
            return_sequences=return_sequences,
            return_state=return_state,
            stateful=stateful,
        )

    def build(self, input_shape):
        input_dim = input_shape[1][-1]
        self.attention_score = AttentionScore(input_dim)

    def call(self, inputs, masks=None):
        attention_weights = self.attention_score(inputs, masks=masks)
        res = self.rnn((inputs[1], attention_weights), masks=masks)
        if self.return_attention_weights:
            return res, attention_weights
        return res

    def get_config(self):
        config = super(AUGRU, self).get_config()
        config.update({
            "units": self.units,
            "return_attention_weights": self.return_attention_weights,
            "return_sequences": self.return_sequences,
            "return_state": self.return_state,
            "stateful": self.stateful
        })
        return config
