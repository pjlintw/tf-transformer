import tensorflow as tf

from encoder import Encoder
from decoder import Decoder

from pprint import pprint


class Transformer(tf.keras.Model):
    """Define computational graph for Transformer

    See "Attention is All You Need"
    """
    def __init__(self, num_layers, d_model, num_head, intermediate_dim, 
                 input_vocab_size, target_vocab_size, pe_input, pe_target,
                 rate=0.1, scope='transformer'):
        super(Transformer, self).__init__()
        self.encoder = Encoder(num_layers, d_model, num_head, intermediate_dim,
                                input_vocab_size, pe_input, rate)
        self.decoder = Decoder(num_layers, d_model, num_head, intermediate_dim,
                                target_vocab_size, pe_target, rate)
        self.emb = tf.keras.layers.Embedding(input_vocab_size, d_model)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size)


    def call(self, inp, tar, training, enc_padding_mask,
             look_ahead_mask, dec_padding_mask):
        enc_output = self.encoder(inp, training, enc_padding_mask)

        dec_output, attention_weights = self.decoder(tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        
        final_output = self.final_layer(dec_output)

        # return final_output
        return final_output, attention_weights