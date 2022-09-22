import json
import os
import tensorflow as tf
from tensorflow_unicorn.layers.transformers import *


def get_config(filename):
    with open(filename, "r") as f:
        config = json.load(f)
    return config


def digest_config(instance, config):
    for key in config:
        setattr(instance, key, config[key])


class BertPath:
    def __init__(self, dirname):
        self.dirname = dirname
        files = os.listdir(dirname)

        if "bert_config.json" in files:
            self.config = os.path.join(dirname, "bert_config.json")
        else:
            for file in files:
                if "config" in file:
                    self.config = os.path.join(dirname, file)
                    break
            else:
                raise FileNotFoundError(f"No configuration file in {dirname}.")

        if "vocab.txt" in files:
            self.vocab = os.path.join(dirname, "vocab.txt")
        else:
            for file in files:
                if "vocab" in file:
                    self.vocab = os.path.join(dirname, file)
                    break
            else:
                raise FileNotFoundError(f"No vocabulary file in {dirname}.")

        self.checkpoint = os.path.join(dirname, "bert_model.ckpt")


class BertModel(tf.keras.Model):

    BERT_BASE_CONFIG = {
        'hidden_size': 768, 'hidden_act': "gelu",
        'initializer_range': 0.02, 'vocab_size': 30522,
        'hidden_dropout_prob': 0.1, 'num_attention_heads': 12,
        'type_vocab_size': 2, 'max_position_embeddings': 512,
        'num_hidden_layers': 12, 'intermediate_size': 3072,
        'attention_probs_dropout_prob': 0.1,
    }

    def __init__(self, config=None, name="bert"):
        super(BertModel, self).__init__(name=name)
        if config is None:
            config = self.BERT_BASE_CONFIG
        self._config = config
        digest_config(self, config)

        self.composite_embeddings = CompositeEmbedding(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            initializer_range=self.initializer_range,
            max_position_embeddings=self.max_position_embeddings,
            mask_zero=False,
            name="embeddings"
        )

        self.encoder = []
        for i in range(self.num_hidden_layers):
            self.encoder.append(EncoderBlock(
                hidden_size=self.hidden_size, hidden_act=self.hidden_act,
                initializer_range=self.initializer_range, hidden_dropout_prob=self.hidden_dropout_prob,
                num_attention_heads=self.num_attention_heads, intermediate_size=self.intermediate_size,
                attention_probs_dropout_prob=self.attention_probs_dropout_prob, name=f"encoder/layer_{i}"
            ))

    def call(self, inputs, training=None):
        if not tf.nest.is_nested(inputs):
            inputs = [inputs, tf.zeros_like(inputs, dtype=tf.int32)]

        x = self.composite_embeddings(inputs)
        for block in self.encoder:
            x = block(x, training=training)

        return x


def get_bert(hidden_size=768, hidden_act="gelu",
             initializer_range=0.02, vocab_size=30522,
             hidden_dropout_prob=0.1, num_attention_heads=12,
             type_vocab_size=2, max_position_embeddings=512,
             num_hidden_layers=12, intermediate_size=3072,
             attention_probs_dropout_prob=0.1,
             extract_all=False, name="bert", **kwargs):
    layer_input = (
        tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                              name=name + "/token_ids_input"),
        tf.keras.layers.Input(shape=(None,), dtype=tf.int32,
                              name=name + "/segment_ids_input")
    )
    composite_embedding = CompositeEmbedding(
        vocab_size=vocab_size, hidden_size=hidden_size,
        type_vocab_size=type_vocab_size, initializer_range=initializer_range,
        max_position_embeddings=max_position_embeddings, name=name + "/embeddings"
    )(layer_input)

    output = composite_embedding
    for i in range(num_hidden_layers):
        if extract_all:
            output = get_encoder_block(
                output,
                hidden_size=hidden_size, hidden_act=hidden_act,
                initializer_range=initializer_range, hidden_dropout_prob=hidden_dropout_prob,
                num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob, name=name + f"/encoder/layer_{i}"
            )
        else:
            output = EncoderBlock(
                hidden_size=hidden_size, hidden_act=hidden_act,
                initializer_range=initializer_range, hidden_dropout_prob=hidden_dropout_prob,
                num_attention_heads=num_attention_heads, intermediate_size=intermediate_size,
                attention_probs_dropout_prob=attention_probs_dropout_prob, name=name + f"/encoder/layer_{i}"
            )(output)

    return tf.keras.Model(inputs=layer_input, outputs=output, name=name)
