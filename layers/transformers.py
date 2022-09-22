import tensorflow as tf
from tensorflow import keras


def get_encoder_block(layer_input,
                      hidden_size=768, hidden_act="gelu",
                      initializer_range=0.02, hidden_dropout_prob=0.1,
                      num_attention_heads=12, intermediate_size=3072,
                      attention_probs_dropout_prob=0.1,
                      name="encoder"):
    attention_output = SelfAttention(
        hidden_size=hidden_size, num_attention_heads=num_attention_heads,
        attention_probs_dropout_prob=attention_probs_dropout_prob, hidden_dropout_prob=hidden_dropout_prob,
        initializer_range=initializer_range,
        name=name + "/attention"
    )(layer_input)
    if hidden_dropout_prob > 0.0:
        attention_output = keras.layers.Dropout(
            hidden_dropout_prob, name=name + "/attention/output/dropout"
        )(attention_output)
    attention_output = keras.layers.LayerNormalization(
        name=name + "/attention/output/LayerNorm", epsilon=keras.backend.epsilon()**2
    )(layer_input + attention_output)

    intermediate_layer = keras.layers.Dense(
        intermediate_size, activation=hidden_act,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range),
        name=name + "/intermediate/dense"
    )(attention_output)

    output = keras.layers.Dense(
        hidden_size,
        kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range),
        name=name + "/output/dense"
    )(intermediate_layer)
    if hidden_dropout_prob > 0.0:
        output = keras.layers.Dropout(hidden_dropout_prob, name=name + "/output/dropout")(output)
    output = keras.layers.LayerNormalization(
        name=name + "/output/LayerNorm", epsilon=keras.backend.epsilon()**2
    )(output + attention_output)
    return output


class CompositeEmbedding(keras.layers.Layer):
    def __init__(self, vocab_size,
                 hidden_size=768, type_vocab_size=2,
                 initializer_range=0.02, hidden_dropout_prob=0.1,
                 max_position_embeddings=512, use_token_type_embeddings=True,
                 use_layer_norm=True, mask_zero=True, **kwargs):
        super(CompositeEmbedding, self).__init__(**kwargs)
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._type_vocab_size = type_vocab_size
        self._initializer_range = initializer_range
        self._hidden_dropout_prob = hidden_dropout_prob
        self._max_position_embeddings = max_position_embeddings
        self._use_token_type_embeddings = use_token_type_embeddings
        self._use_layer_norm = use_layer_norm
        self._mask_zero = mask_zero

        if self._hidden_dropout_prob > 0:
            self._dropout = keras.layers.Dropout(hidden_dropout_prob, name="embedding_dropout")
        else:
            self._dropout = None

        if self._use_layer_norm:
            self._ln = keras.layers.LayerNormalization(name="LayerNorm", epsilon=keras.backend.epsilon()**2)
        else:
            self._ln = None

    def build(self, input_shape):
        # with tf.name_scope(self.name):
        self.word_embeddings = self.add_weight(
            name="word_embeddings",
            shape=(self._vocab_size, self._hidden_size),
            dtype=tf.float32,
            initializer=keras.initializers.TruncatedNormal(stddev=self._initializer_range)
        )
        if self._use_token_type_embeddings:
            self.token_type_embeddings = self.add_weight(
                name="token_type_embeddings",
                shape=(self._type_vocab_size, self._hidden_size),
                dtype=tf.float32,
                initializer=keras.initializers.TruncatedNormal(stddev=self._initializer_range)
            )
        self.position_embeddings = self.add_weight(
            name="position_embeddings",
            shape=(self._max_position_embeddings, self._hidden_size),
            dtype=tf.float32,
            initializer=keras.initializers.TruncatedNormal(stddev=self._initializer_range)
        )
        self.built = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            return mask
        elif self._mask_zero:
            if tf.nest.is_nested(inputs):
                inputs = inputs[0]
            return tf.not_equal(inputs, 0)
        else:
            return None

    def call(self, inputs, training=None):
        if self._use_token_type_embeddings:
            x = tf.nn.embedding_lookup(self.word_embeddings, inputs[0]) + \
                tf.nn.embedding_lookup(self.token_type_embeddings, inputs[1]) +\
                self.position_embeddings[tf.newaxis, :tf.shape(inputs[0])[1], :]
        else:
            if tf.nest.is_nested(inputs):
                inputs = inputs[0]
            x = tf.nn.embedding_lookup(self.word_embeddings, inputs) + \
                self.position_embeddings[tf.newaxis, :tf.shape(inputs)[1], :]

        if self._dropout is not None:
            x = self._dropout(x, training=training)

        if self._ln is not None:
            x = self._ln(x)
        return x

    def compute_output_shape(self, input_shape):
        if tf.nest.is_nested(input_shape):
            return input_shape[0].concatenate(self._hidden_size)
        else:
            return input_shape.concatenate(self._hidden_size)

    def get_config(self):
        config = super(CompositeEmbedding, self).get_config()
        config.update({
            "vocab_size": self._vocab_size,
            "hidden_size": self._hidden_size,
            "type_vocab_size": self._type_vocab_size,
            "initializer_range": self._initializer_range,
            "hidden_dropout_prob": self._hidden_dropout_prob,
            "max_position_embeddings": self._max_position_embeddings,
            "use_token_type_embeddings": self._use_token_type_embeddings,
            "use_layer_norm": self._use_layer_norm,
            "mask_zero": self._mask_zero
        })
        return config


class SelfAttention(keras.layers.Layer):
    def __init__(self, hidden_size=768, num_attention_heads=12,
                 attention_probs_dropout_prob=0.1,
                 initializer_range=0.02, use_bias=True,
                 **kwargs):
        super(SelfAttention, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._num_attention_heads = num_attention_heads
        self._attention_probs_dropout_prob = attention_probs_dropout_prob
        self._initializer_range = initializer_range
        self._use_bias = use_bias
        self._size_per_head = self._hidden_size // self._num_attention_heads

        self._query_dense = keras.layers.Dense(
            hidden_size, kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="self/query", use_bias=self._use_bias
        )
        self._key_dense = keras.layers.Dense(
            hidden_size, kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="self/key", use_bias=self._use_bias
        )
        self._value_dense = keras.layers.Dense(
            hidden_size, kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="self/value", use_bias=self._use_bias
        )
        self._output_dense = keras.layers.Dense(
            hidden_size, kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="output/dense", use_bias=self._use_bias
        )

        if attention_probs_dropout_prob > 0.0:
            self._dropout = keras.layers.Dropout(attention_probs_dropout_prob, name="self/dropout")
        else:
            self._dropout = None

        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        q = self._query_dense(inputs)
        k = self._key_dense(inputs)
        v = self._value_dense(inputs)

        seq_len = tf.shape(inputs)[1]

        q = tf.reshape(q, (tf.shape(inputs)[0], seq_len, self._num_attention_heads, self._size_per_head))
        q = tf.transpose(q, [0, 2, 1, 3])

        k = tf.reshape(k, (tf.shape(inputs)[0], seq_len, self._num_attention_heads, self._size_per_head))
        k = tf.transpose(k, [0, 2, 1, 3])

        v = tf.reshape(v, (tf.shape(inputs)[0], seq_len, self._num_attention_heads, self._size_per_head))
        v = tf.transpose(v, [0, 2, 1, 3])
        # Shape of q, k, v: (batch_size, num_heads, seq_len, size_per_head)

        score_mat = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self._size_per_head, dtype=tf.float32))
        # Shape of score_mat: (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            mask = tf.cast(tf.logical_not(mask), tf.float32) * -1e4
            score_mat += mask[:, tf.newaxis, :, tf.newaxis]
        attention_weights = tf.nn.softmax(score_mat)

        if self._dropout is not None:
            attention_weights = self._dropout(attention_weights, training=training)
        else:
            attention_weights = self.attention_weights

        attention_output = tf.reduce_sum(
            attention_weights[..., tf.newaxis] * v[:, :, tf.newaxis, :, :], axis=-2
        )
        # Shape of attention output: (batch_size, num_heads, seq_len, size_per_head)

        attention_output = tf.transpose(attention_output, [0, 2, 1, 3])
        attention_output = tf.reshape(
            attention_output, (tf.shape(attention_output)[0], tf.shape(attention_output)[1], self._hidden_size)
        )
        # Shape of attention output: (batch_size, seq_len, hidden_size)

        output = self._output_dense(attention_output)
        return output

    def get_config(self):
        config = super(SelfAttention, self).get_config()
        config.update({
            "hidden_size": self._hidden_size,
            "num_attention_heads": self._num_attention_heads,
            "attention_probs_dropout_prob": self._attention_probs_dropout_prob,
            "initializer_range": self._initializer_range,
            "use_bias": self._use_bias,
        })
        return config


class EncoderBlock(keras.layers.Layer):
    def __init__(self, hidden_size=768, hidden_act="gelu",
                 initializer_range=0.02, hidden_dropout_prob=0.1,
                 num_attention_heads=12, intermediate_size=3072,
                 attention_probs_dropout_prob=0.1,
                 **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self._hidden_size = hidden_size
        self._hidden_act = hidden_act
        self._initializer_range = initializer_range
        self._hidden_dropout_prob = hidden_dropout_prob
        self._num_attention_heads = num_attention_heads
        self._intermediate_size = intermediate_size
        self._attention_probs_dropout_prob = attention_probs_dropout_prob

        self._self_attention = SelfAttention(
            hidden_size=hidden_size, num_attention_heads=num_attention_heads,
            attention_probs_dropout_prob=attention_probs_dropout_prob, initializer_range=initializer_range,
            name="attention"
        )

        if hidden_dropout_prob > 0.0:
            self._dropout1 = keras.layers.Dropout(hidden_dropout_prob, name="attention/output/dropout")
        self._ln1 = keras.layers.LayerNormalization(
            name="attention/output/LayerNorm", epsilon=keras.backend.epsilon() ** 2
        )

        self._intermediate_dense = keras.layers.Dense(
            intermediate_size, activation=hidden_act,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="intermediate/dense"
        )

        self._output_dense = keras.layers.Dense(
            hidden_size,
            kernel_initializer=keras.initializers.TruncatedNormal(stddev=initializer_range),
            name="output/dense"
        )
        if hidden_dropout_prob > 0.0:
            self._dropout1 = keras.layers.Dropout(hidden_dropout_prob, name="attention/output/dropout")
            self._dropout2 = keras.layers.Dropout(hidden_dropout_prob, name="output/dropout")
        else:
            self._dropout1 = None
            self._dropout2 = None

        self._ln2 = keras.layers.LayerNormalization(name="output/LayerNorm", epsilon=keras.backend.epsilon() ** 2)

        self.supports_masking = True

    def call(self, inputs, mask=None, training=None):
        attention_output = self._self_attention(inputs, mask=mask, training=training)
        if self._dropout1 is not None:
            attention_output = self._dropout1(attention_output)
        attention_output = self._ln1(attention_output + inputs)

        intermediate = self._intermediate_dense(attention_output)
        output = self._output_dense(intermediate)
        if self._dropout2 is not None:
            output = self._dropout2(output, training=training)
        output = self._ln2(output + attention_output)
        return output

    def get_config(self):
        config = super(EncoderBlock, self).get_config()
        config.update({
            "hidden_size": self._hidden_size,
            "hidden_act": self._hidden_act,
            "initializer_range": self._initializer_range,
            "hidden_dropout_prob": self._hidden_dropout_prob,
            "num_attention_heads": self._num_attention_heads,
            "intermediate_size": self._intermediate_size,
            "attention_probs_dropout_prob": self._attention_probs_dropout_prob
        })
        return config
