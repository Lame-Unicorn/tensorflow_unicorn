import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.engine.data_adapter import unpack_x_y_sample_weight


class MLMTrainer(keras.Model):
    def __init__(self, model, return_original_output=False, **kwargs):
        super(MLMTrainer, self).__init__(**kwargs)
        self._return_original_output = return_original_output
        self.model = model

    def build(self, input_shape):
        self.model.build(input_shape)
        self.word_embeddings = self.model.composite_embeddings.word_embeddings
        self.built = True

    def call(self, inputs, training=None):
        model_output = self.model(inputs, training=training)
        sim_mat = tf.tensordot(model_output, self.word_embeddings, [[-1], [-1]])
        # shape of sim_mat: (batch_size, seq_len, vocab_size)
        if self._return_original_output:
            return model_output, sim_mat
        else:
            return sim_mat


class SimCSETrainer(keras.Model):
    def __init__(self, model, temperature=0.05, pooling="class", embed_dim=128, **kwargs):
        super(SimCSETrainer, self).__init__(**kwargs)
        self._temperature = temperature
        self._pooling = pooling
        self.model = model
        if pooling == "class":
            self.pooler = keras.layers.Lambda(lambda x: x[:, 0], name="class_pooling")
        elif pooling == "average":
            self.pooler = keras.layers.GlobalAveragePooling1D(name="average_pooling")
        self.decomposition = keras.layers.Dense(embed_dim, activation=None)

    def call(self, inputs, training=None):
        embed1 = self.decomposition(self.pooler(self.model(inputs, training=training)))
        embed2 = self.decomposition(self.pooler(self.model(inputs, training=training)))
        sim_mat = -keras.losses.cosine_similarity(embed1[:, tf.newaxis], embed2[tf.newaxis, ...]) / self._temperature
        return sim_mat


class EmbeddingFGSMWrapper(keras.Model):
    def __init__(self, model, embeddings, epsilon=0.1, **kwargs):
        if not hasattr(model, "loss"):
            raise AttributeError(f"Your model {model.name} should be compile with a loss.")

        super(EmbeddingFGSMWrapper, self).__init__(**kwargs)
        self.model = model
        self.embeddings = embeddings
        self.epsilon = epsilon
        
        self.compile(
            loss=self.model.loss,
            optimizer=self.model.optimizer
        )

    @tf.function
    def train_step(self, data):
        x, y, sample_weight = unpack_x_y_sample_weight(data)

        with tf.GradientTape() as tape:
            y_pred = self.model(x)
            loss = self.compiled_loss(
                y, y_pred, sample_weight, regularization_losses=self.model.losses)
        g = tf.zeros_like(self.embeddings, dtype=tf.float32) + tape.gradient(loss, self.embeddings)

        self.embeddings.assign_add(self.epsilon * g)
        res = self.model.train_step(data)
        self.embeddings.assign_sub(self.epsilon * g)

        return res

    def call(self, inputs, mask=None, training=None):
        return self.model(inputs, mask=mask, training=training)
