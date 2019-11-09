from nn import attention, ffn, create_positional_embedding

import tensorflow as tf

LAYER_NORM_BIAS_DEFAULT_NAME = "ln_bias"
LAYER_NORM_GAIN_DEFAULT_NAME = "ln_gain"
LAYER_NORMALIZATION_DEFAULT_NAME = "layer_normalization"



class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_head, intermediate_dim, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.d_model = d_model
        self.num_head = num_head

        assert d_model % num_head == 0

        self.head_size = d_model // num_head
        self.intermediate_dim = intermediate_dim
        self.rate = rate

        self.first_droput = tf.keras.layers.Dropout(rate)
        self.second_dropout = tf.keras.layers.Dropout(rate)
        self.third_droput = tf.keras.layers.Dropout(rate)

    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        attn_1, attn_1_weights = attention(inp_query=x,
                                           inp_key=x,
                                           inp_value=x,
                                           num_head=self.num_head,
                                           head_size=self.head_size,
                                           attention_mask=look_ahead_mask)

        out_1 = self.first_droput(attn_1, training=training)
        # out1 = tf.contrib.layers.layer_norm(x + out_1)

        attn_2, attn_2_weights = attention(inp_query=out_1, 
                                           inp_key=enc_output,
                                           inp_value=enc_output,
                                           num_head=self.num_head,
                                           head_size=self.head_size,
                                           attention_mask=padding_mask)

        attn_2 = self.second_dropout(attn_2, training=training)
        # out2 = tf.contrib.layers.layer_norm(out1 + attn_2)

        ffn_out = ffn(attn_2)
        ffn_out = self.third_droput(ffn_out, training=training)
        #last_output = tf.contrib.layers.layer_norm( out2 + ffn_out)
        last_output = tf.contrib.layers.layer_norm(ffn_out)

        return last_output, attn_1_weights, attn_2_weights



class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_head, intermediate_dim, target_vocab_size,
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = create_positional_embedding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_head, intermediate_dim, rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)


    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]

        # collect attention weights
        attention_weights = {}

        x = self.embedding(x)
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output=enc_output, training=training, 
                                                 look_ahead_mask=look_ahead_mask, padding_mask=padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        return x, attention_weights