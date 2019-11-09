from nn import attention, ffn, create_positional_embedding

import numpy as np
import tensorflow as tf


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_head, intermediate_dim, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.d_model = d_model
        self.num_head = num_head

        assert d_model % num_head == 0

        self.head_size = d_model // num_head
        self.intermediate_dim = intermediate_dim
        self.rate = rate

        self.first_dropout = tf.keras.layers.Dropout(self.rate)
        self.second_dropout = tf.keras.layers.Dropout(self.rate)


    def call(self, x, training, mask):

        attn_output, _ = attention(inp_query=x,
                                   inp_key=x,
                                   inp_value=x,
                                   num_head=self.num_head,
                                   head_size=self.head_size,
                                   attention_mask=mask)
        attn_output = self.first_dropout(attn_output, training=training)
        first_norm_out = tf.contrib.layers.layer_norm(x + attn_output)

        ffn_output = ffn(first_norm_out)
        ffn_output = self.second_dropout(ffn_output, training=training)
        second_norm_out = tf.contrib.layers.layer_norm(first_norm_out + ffn_output)

        return second_norm_out


class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_head, intermediate_dim, input_vocab_size,
                maximum_position_encoding, rate=0.1):

        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = create_positional_embedding(maximum_position_encoding, d_model)

        self.enc_layers = [EncoderLayer(d_model, num_head, intermediate_dim, rate)
                           for _ in range(num_layers)]


        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):

        seq_len = tf.shape(x)[1]

        #x = tf.convert_to_tensor(x, np.float32)
        x = self.embedding(x)

        # x /= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len , :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x