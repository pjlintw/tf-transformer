"""reimplement `attention is all you need`, inspired by `https://kexue.fm/archives/4765/comment-page-1#comments`

"""
import numpy as np

import tensorflow as tf

def create_initializer(initializer_range=0.02):
    """Create a `truncated_normal_initializer` with the given range.

    For truncated normal distribution, values are dropped and re-pick.
    if magnitude is more than 2 standard deviations from mean. 

    Point for using truncated normal is to aviod sauration of tome
    functions (sigmoid etc). When the weights are too large, output
    of tome function trend to be large as well. Thus the gradient
    is too small, slowing down learning

    * This function borrow from bert/modeling.py
    """
    return tf.truncated_normal_initializer(stddev=initializer_range)

def dense(inputs, output_size, bias=True, scope='dense'):
    """Project input tensor to query, key, value and concated scaled
    dot-product attention

    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # input_size (batch_size, sequence_maxlen, dim)
        #input_size = tf.shape(inputs)[-1]
        input_size = int(inputs.shape[-1])   # dim = 50
        #print('input shape', inputs.shape)
        
        #  shape = (dim, output_size)
        W = tf.Variable(tf.random_uniform([input_size, output_size], -0.05, 0.05))

        if bias:
            b = tf.Variable(tf.random_uniform([output_size], -0.05, 0.05))
        else:
            b = 0
        
        # shape (batch_size * sequence_maxlen,  dim)
        x = tf.reshape(inputs, [-1, input_size])

        # query = input_embedding * W, 
        outputs = tf.matmul(x, W) + b

        # shape (batch_size, sequence_maxlen , output_size)
        #outputs = tf.reshape(outputs, [-1, inputs.shape[-2], output_size])
        outputs = tf.reshape(outputs, \
                             tf.concat([tf.shape(inputs)[:-1], [output_size]], 0)
                            )
    return outputs


def attention(inp_query, inp_key, inp_value, num_head, head_size, attention_mask=None, scope='attention'):
    """Implement of `scaled dot-product` and `multi-head attention`.

    This function computes `scaled dot product` and `multi-head attention`.
    
    formula:
        (get Q, K, V): dense(inputs)
        (scaled dot-product attention): softmax((Q * transpose(K)) / square_root(d_k) ) * V
        (multi-head attention): 

    Q, K, V: each has shape (batch_size, sequence_maxlen, num_head * head_size)
    Scaled dot-product attion: multi-head-attention(Q, K, V) = softmax( Q * trans(K) / sqrt(dk) ) * V
    For details, See "Attention Is All You Need"

    Args:
        inputs: shape (batch_size, sequence_maxlen, dim)

    Return:
        concated_output: shape [batch_size, seq_len, num_head * head_size]

    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        ### 1. Query, Key and Value projection ###
        # shape: (batch_size, sequence_maxlen, num_head * head_size)
        Q = dense(inp_query, num_head * head_size, False)
        # split last dimension to num_head * head_size
        Q = tf.reshape(Q, (-1, tf.shape(Q)[1], num_head, head_size))
        # change `sequence_maxlen` and `num_head`
        # shape (batch_size, num_head, sequence_maxlen, head_size)
        Q = tf.transpose(Q, [0, 2, 1, 3])

        K = dense(inp_key, num_head * head_size, False)
        K = tf.reshape(K, (-1, tf.shape(K)[1], num_head, head_size))
        K = tf.transpose(K, [0, 2, 1, 3])

        V = dense(inp_value, num_head * head_size, False)
        V = tf.reshape(V, (-1, tf.shape(V)[1], num_head, head_size))
        V = tf.transpose(V, [0, 2, 1, 3])

        ### 2. attention_score(Q, trans(K)) ###
        # Q * transpose(K) = (batch_size, num_head, sequence_maxlen, head_dim) * (batch_size, num_head, head_dim, sequence_maxlen) 
        #                    
        # attention_score: shape (batch_size, num_head, sequence_maxlen_q, sequence_maxlen_k)
        attention_score = tf.matmul(Q, K, transpose_b=True)

        ### 3. scaled score ###
        scaled_att_score = attention_score / tf.sqrt(float(head_size))

        ### 4. mask ###
        if attention_mask is not None:
            scaled_att_score += attention_mask
            #scaled_att_score = mask(scaled_att_score, seq_len=seq_len, mode='add')
        
        ### 5. scaled_attention ###
        # scaled_att_score (batch_size, num_head, sequence_maxlen_q, sequence_maxlen_k)
        # softmax(attention_score)
        attention_weight = tf.nn.softmax(scaled_att_score, axis=-1)

        # (batch, num_h, seq_len_q, seq_len_k) * (batch, num_h, seq_len_k, head_size)
        # scaled_attention shape (batch_size, num_head, sequence_maxlen_q, head_size)
        scaled_attention = tf.matmul(attention_weight, V)

        # reverse `num_head` and `sequence_maxlen`
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3])
        
        # reshape tensor as (batch_size, sequence_maxlen, num_head * head_size) for last dense layer
        concat_sum_weighted_value = tf.reshape(scaled_attention, 
                                               (-1, tf.shape(scaled_attention)[1], num_head*head_size))

        # final dense layer
        last_dense_output = dense(inputs=concat_sum_weighted_value,  
                                   output_size=num_head * head_size,
                                   bias=True)
        
    return last_dense_output, attention_weight


def create_positional_embedding(num_position, d_model):
    """Create positonal embeddings with shape (num_position, d_model)

    Given maximum position and d_model, creating embedding and slicing it

    Args:
      eko
      
    Return:
        

    """
    
    pos = np.arange(num_position)[:, np.newaxis]
    i = np.arange(d_model)[np.newaxis:]

    angle_rates = 1 / np.power(10000, (2 * (i //2)) / np.float32(d_model))
    angle_rads = pos * angle_rates

    # apply sin to even
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return  tf.cast(pos_encoding, dtype=tf.float32)


def postional_embedding(inputs, pos_dim):
    """Create absolute positional embedding 

    x : shape (batch_size, seq_maxlen, dims)

    Returns:
        position_embedding : shape (batch_size, seq_maxlen, pos_dim)

    """
    #   1/ 10000 ^ (2 * i * (1/2) / pos_dim )
    # shape (seq_max, )
    position_j = 1. / tf.pow(10000., 2 * tf.range(pos_dim/2 ,dtype=tf.float32) / pos_dim)

    # shape (1, seq_mqxlen)
    position_j = tf.expand_dims(position_j, 0)
    #print(position_j.shape)

    # shape (batch_size, seq_maxlen, dims)
    batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]

    # sequence of indices
    # shape (seq_maxlen, )
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)

    position_i = tf.expand_dims(position_i, 1)

    position_ij = tf.matmul(position_i, position_j)
    
    # (batch_size, pos_dim)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)

    position_embedding = tf.expand_dims(position_ij, 0) + tf.zeros((batch_size, seq_len, pos_dim))

    return position_embedding


def mask(inputs, seq_len, mode='add'):
    """Implement padding mask. It cauculate by given 1D tensor which
    indicate lengths of sentence in a batch

    Args
        inputs: shape(batch_size, num_heads, sequence_maxlen, sequence_maxlen)
        seq_len: 1D tensor with batch_size, element is length of each sentence
                [59, 47, 59, 29, 88, 64, ...]

        mode: {mul, add}
             mul: set extra token with 0, use it beforce fully connected layer

             add: add a great negatve value to padding token, use it beforce softmax
                 because softmax will set it 0

    Returns
        masked_output: shape(batch_size, num_heads, seq_maxlen, seq_maxlen)
    """
    batch_size = tf.shape(inputs)[0]
    batch_size = tf.shape(inputs)[-1]
    #seq_maxlen = int(inputs.shape[-1])

    # Create mask matrix with bool
    # `tf.sequence_mask([1, 3, 2], maxlen=5)`  # [[True, False, False, False, False],
    #                                          #  [True, True, True, False, False],
    #                                          #  [True, True, False, False, False]]
    #
    # lengths: [seq_len1, seq_len2, ..., seq_len_batch_size ]
    # maxlen: seq_maxlen
    # boolean Tensor with shape (batch_size, seq_maxlen)
    #        (batch_size, ) --> (batch_size, seq_maxlen)
    mask_seq = tf.sequence_mask(lengths=seq_len, maxlen=seq_maxlen)

    # cast it to 1 or 0
    # shape (batch_size, seq_maxlen)
    #         [[1. 0.]]
    #          [1. 0.]]
    mask_seq = tf.cast(mask_seq, tf.float32)
    
    # expand dimension at `dims` postion
    # shape: (batch_size, seq_maxlen) --> (shape, 1, seq_maxlen)
    #         [[[1.]
    #           [0.]]
    #
    #           [1.]
    #           [0.]]]
    #  
    # shape (batch_size, 1, seq_maxlen)
    mask_seq = tf.expand_dims(mask_seq, 1)

    # shape (batch_size, seq_maxlen, 1)
    ones = tf.ones([batch_size, seq_maxlen, 1])
    
    ### broadcast ###
    # example: batch: 1, seq_len: 2, seq_maxlen: 10
    #
    #                          seq_max_len: 10
    #                 [[1, 1, 0, 0, 0, 0, 0, 0 ,0 ,0 ],
    # seq_max_len:10   [1, 1, 0, 0, 0, 0, 0, 0 ,0 ,0 ],
    #                  [1, 1, 0, 0, 0, 0, 0, 0 ,0 ,0 ],
    #                    .......
    #                  [1, 1, 0, 0, 0, 0, 0, 0 ,0 ,0 ]]
    # shape (batch_size, seq_maxlen, seq_maxlen)
    mask_seq = mask_seq * ones

    # shape (batch_size, seq_maxlen, seq_max_len) --> (batch_size, 1, seq_maxlen, seq_max_len) 
    attn_mask = tf.expand_dims(mask_seq, axis=[1])

    # mask
    if mode == 'mul':
        masked_output = inputs * mask_seq
    elif mode == 'add':
        adder = (1.0 - tf.cast(attn_mask, tf.float32)) * 1e12
        masked_output = inputs - adder
    
    return masked_output

### MASK ###
#import numpy as np

# (batch_size, num_head, max_len, maxlen)
# fake_x = np.array([
#                     [
#                       [
#                         [1,1,1], [1,1,1], [1,1,1]
#                                                   ],
#                       [
#                         [1,1,1], [1,1,1], [1,1,1]
#                                                   ]],
#                     [
#                       [
#                         [1,1,1], [1,1,1], [1,1,1]
#                                                   ],
#                       [
#                         [1,1,1], [1,1,1], [1,1,1]
#                                                   ],
#                             ]])

# fake_seq_len = np.array([2, 1])

# mask_seq = mask(fake_x, seq_len=fake_seq_len)

# sess = tf.Session()
# print(sess.run(mask_seq))

## MASK ###

def layer_norm(inputs):
    """add & norm """
    norm = tf.contrib.layers.layer_norm(inputs)
    return norm


def ffn(input_tensor, intermediate_dim=2048 , use_non_negative=True, scope='feed_forward'):
    """Implement of feed forward network

    FFN(x) = dense(relu(dense(x)))

    Args:
      input_tensor: shape (batch_size, sequence_maxlen, dim)
      intermediate_dims: paper provides 4 times of input_tensor
      use_non_negative: (optional) boolean, set all negative numbers to zero if true, otherwise don't change number at all

    Returns:
      ffn_output: shape (batch_size, sequence_maxlen, dim)
    """
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        output_dims = int(input_tensor.shape[-1])

        intermediate_output = dense(inputs=input_tensor, 
                                    output_size=intermediate_dim)
            
        if use_non_negative:
            intermediate_output = set_non_negative_tensor(input_tensor=intermediate_output,
                                                          threadhold=0)

        ffn_output = dense(inputs=intermediate_output, 
                            output_size=output_dims)

    return ffn_output


def set_non_negative_tensor(input_tensor, threadhold=0):
    """Clip all negatvie number to zero
    """
    output_tensor = tf.clip_by_value(input_tensor, clip_value_min=threadhold, clip_value_max=np.Infinity)
    
    return output_tensor


def mask_future_words(inputs, mode='add'):
    """mask

    Args
        inputs: shape(batch_size, num_heads, sequence_maxlen, sequence_maxlen)
        seq_len: 1D array with batch_size, element is length of each sentence
                [59, 47, 59, 29, 88, 64, ...]

        mode: {mul, add}
             mul: set extra token with 0, use it beforce fully connected layer

             add: add a great negatve value to padding token, use it beforce softmax
                 because softmax will set it 0

    Returns
        masked_output: shape(batch_size, num_heads, seq_maxlen, seq_maxlen)
    """
    batch_size = tf.shape(inputs)[0]
    seq_maxlen = int(inputs.shape[-1])

    # `tf.liang.band_part(inputs, num_lower, num_upper)`
    #  will copy a tensor setting everything outside a central band in martrix
    #
    # num_lower: -1 keep value
    # num_upper: 0 dis
    masked_seq = 1- tf.linalg.band_part(tf.ones((seq_maxlen, seq_maxlen)), -1, 0)
    return masked_seq

    # seq_len = tf.range(1, seq_maxlen+1, 1)
    # seq_mask = tf.sequence_mask(lengths=seq_len, maxlen=seq_maxlen)
    #return tf.cast(seq_mask, dtype=tf.int32)

