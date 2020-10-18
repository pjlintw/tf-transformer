import numpy as np
import functools
from pathlib import Path
import time
import tensorflow as tf

from dataset_utils import batch_generator, input_fn, create_tok2id_from_vocab_file, create_tf_hash_table, convert_idx_to_token_tensor
from transformer import Transformer
np.set_printoptions(threshold=1000)


def convert_tensor_to_string(x, lookup_table, sess):
    """Convert id tensor into string"""
    l = list()
    
    for char_set in x:
      collections = list()
      for char in char_set:
        collections.append(char)
      l.append(collections)

    return l

def _pad_sequence_to_same_length(x, y):
    """Pad x and y as same length"""
    with tf.name_scope("pad_to_same_length"):
        x_length = tf.shape(x)[1]
        y_length = tf.shape(y)[1]

    max_length = tf.maximum(x_length, y_length)

    x = tf.pad(x, [[0, 0], [0, max_length - x_length], [0, 0]])
    y = tf.pad(y, [[0, 0], [0, max_length - y_length]])

    return x, y

def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    with tf.name_scope("loss", values=[logits, labels]):
        logits, labels = _pad_sequence_to_same_length(logits, labels)

    with tf.name_scope("smooth_xent", values=[logits, labels]):
        confidence = 1.0 - smoothing
        low_confidence = (1.0 - confidence) / tf.to_float(vocab_size-1)
        soft_targets = tf.one_hot(
            tf.cast(labels, tf.int32),
            depth=vocab_size,
            on_value=confidence,
            off_value=low_confidence)

        xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=logits, labels=soft_targets)

        # normalizing_constant = -(
        #   confidence * tf.log(confidence) + tf.to_float(vocab_size - 1) *
        #   low_confidence * tf.log(low_confidence + 1e-20))
        # xentropy -= normalizing_constant
    
    # boolean to float
    weights = tf.to_float(tf.not_equal(labels, 0))
    return xentropy * weights, weights

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    #seq = tf.cast(tf.math.equal(seq, '<pad>'), tf.string)

    # add extra dimensions to add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):
  # Encoder padding mask
  enc_padding_mask = create_padding_mask(inp)
  
  # Used in the 2nd attention block in the decoder.
  # This padding mask is used to mask the encoder outputs.
  dec_padding_mask = create_padding_mask(inp)
  
  # Used in the 1st attention block in the decoder.
  # It is used to pad and mask future tokens in the input received by 
  # the decoder.
  look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
  dec_target_padding_mask = create_padding_mask(tar)
  combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
  return enc_padding_mask, combined_mask, dec_padding_mask

def convert_idx_to_token_tensor(x, idx2tok):
    """Convert int32 tensor to string tensor. 

    Args:
      x: 2D int32 tensor
      id2tok: dictionary, map index into token

    Returns:
      2D string tensor

    """
    def py_convert_idx_to_token_fn(x):
        
        print(id2tok[3])
        return [ ([ idx2tok[char] for batch in x for char in batch ] )  ]
    return tf.py_func(py_convert_idx_to_token_fn, [x], tf.string)


def train_step(inp, tar_inp, tar_real, model, sess, target_vocab_size, global_step, ids2tok_talbe, learning_rate_fn=1):
    """Execute prediction and update"""
    # tar_inp = tar[:, :-1]
    # tar_real = tar[:, 1:]

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    predictions, _ = model(inp, tar_inp,
                           True,
                           enc_padding_mask,
                           combined_mask,
                           dec_padding_mask)

    

    y_onehot = tf.one_hot(tf.cast(tar_real, tf.int32), depth=target_vocab_size)
    _label_smooth = ((1- 0.1) * y_onehot) + (0.1 / target_vocab_size )

    cross_ent = tf.nn.softmax_cross_entropy_with_logits_v2(logits=predictions, labels=_label_smooth)
    weights = tf.to_float(tf.not_equal(tar_real, 0))
    loss =  tf.reduce_sum(cross_ent * weights) / (tf.reduce_sum(weights) + 1e-7)

    #xentropy, weights = padded_cross_entropy_loss(predictions, tar_real, 0.1, vocab_size=target_vocab_size)
    #loss = tf.reduce_sum(xentropy) / tf.reduce_sum(weights)

    # y_hat    
    # map pred to index of vocaburary

    pred_ids = tf.to_int32(tf.argmax(predictions, axis=-1))
    out_seq = ids2tok_talbe.lookup(pred_ids)
    
    # 0.0001
    lr = 0.00001

    #print('learning rate', lr)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=global_step)

    tf.summary.scalar('lr', lr)
    tf.summary.scalar('loss', loss)
    #tf.summary.scalar('global_step', global_step)

    summaries = tf.summary.merge_all()
    
    return loss, train_op, global_step, predictions, pred_ids, weights


def trainer(transformer, dataset, num_vocab, num_examples, lookup_table, ids2tok_dict, ids2tok_talbe, sess):

    start = time.time()
    step = 1
    epoch = 1
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # reset loss and acc

    source_ph = tf.placeholder(tf.float32, [None, None])
    tar_inp_ph = tf.placeholder(tf.float32, [None, None])
    tar_real_ph = tf.placeholder(tf.float32, [None, None])

    loss, train_op, _, predictions, pred_ids, mask_m = train_step(source_ph,
                                                                  tar_inp_ph,
                                                                  tar_real_ph, 
                                                                  model=transformer, 
                                                                  sess=sess, 
                                                                  target_vocab_size=num_vocab, 
                                                                  global_step=global_step, 
                                                                  ids2tok_talbe=ids2tok_talbe)    
    sess.run([tf.global_variables_initializer(), ids2tok_table.init])

    ids2tok_dict = dict([ (np.int_(i), ids2tok_dict[i]) for i in ids2tok_dict.keys()])
    for (batch, batch_data) in enumerate(batch_generator(sess, train_dataset)):

        # this is numpy
        ######
        # batch_generator yields batch with numpy array
        ######
        (source, _), (tar_inp, tar_real) = batch_data
        # print(source)
        # print(tar_inp)
        # print(tar_real)

        # np_arr to tensor & lookup
        #source = lookup_table.lookup(tf.convert_to_tensor(source, tf.string))
        #tar_inp = lookup_table.lookup(tf.convert_to_tensor(tar_inp, tf.string))
        #tar_real = lookup_table.lookup(tf.convert_to_tensor(tar_real, tf.string))

        ls, op, gt, pred, ids, mask = sess.run([loss, train_op, global_step, predictions, pred_ids, mask_m], feed_dict={source_ph: source,
                                      tar_inp_ph: tar_inp,
                                      tar_real_ph: tar_real})
        

        print('global step', gt)
        print('loss', ls)

        # variables_names = [v.name for v in tf.trainable_variables()]
        # values = sess.run(variables_names)
        # for k, v in zip(variables_names, values):
        #     print("Variable: ", k)
        #     print("Shape: ", v.shape)
            
        # check predictions
        

        # print('source')
        # for batch in source:
        #     for each_tok in batch:
        #         print(ids2tok_dict[each_tok] ,end=' ')
        #     print()    
        #     break
      
        # print('target')
        # for batch in tar_real:
        #     for each_tok in batch:
        #         print(ids2tok_dict[each_tok] ,end=' ')
        #     print()    
        #     break
        if gt % 30 == 0:
            for batch in ids:
                for each_tok in batch:
                    print(ids2tok_dict[each_tok] ,end=' ')
                print()
                break



        if (step*2) % num_examples == 0:
            print(f'Time taken for 1 epoch: {time.time() - start} secs\n')
            # add epoch
            epoch += 1

            # reset start
            start = time.time()

        if step % 5 == 0:
            pass
        step+=1
          
        # if batch % 50 == 0:
        #     pass
        #     # log out loss

        #     if (epoch + 1) % 5 == 0:
        #         pass
        #     # save checkpoint

        # print(f'Saving checkpoint for epoch {epoch}')



if __name__ == '__main__':
    data_dir = ''
    model_dir = ''
    
    # hparams   
    EPOCHS = 2000
    batch_size = 32
    global_step = 0
    current_epoch = 1
    stop_if_no_increase = 5
    

    tok2id, id2tok = create_tok2id_from_vocab_file('./data/vocab.txt')
    print(tok2id)
    print(id2tok)

    #print(tok2id)
    tok_lst = list(tok2id.keys())
    ids_lst = [tok2id[k] for k in tok_lst]
    oov_idx = len(tok_lst) + 1

    #print(oov_idx)
    tok2ids_table = create_tf_hash_table(keys=tok_lst,
                                         values=ids_lst,
                                         key_dtype=tf.string, 
                                         value_dtype=tf.int32,
                                         default_value=oov_idx)

    ids2tok_table = create_tf_hash_table(keys=ids_lst,
                                         values=tok_lst,
                                         key_dtype=tf.int32, 
                                         value_dtype=tf.string,
                                         default_value='<UNK>')

    # init transformer
    transformer = Transformer(num_layers=1, d_model=128, num_head=8, intermediate_dim=300, 
                                input_vocab_size=oov_idx-1, target_vocab_size=oov_idx-1, 
                                pe_input=100, pe_target=100, rate=0.1)

    import time
    s = time.time()
    # fetch dataset
    train_dataset, num_examples = input_fn('train', 'train', tok2id, num_epoch=EPOCHS, batch_size=32, shuffle=False)

    t_create_dataset = time.time()-s
    print(f'Taken {t_create_dataset} for creating ')

    #x = tf.convert_to_tensor(np.array(['dsjkdaosdjiasjdisad']), tf.string)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run([tok2ids_table.init])
        #print(sess.run(tok2ids_table.lookup(x)))
        
        trainer(transformer=transformer, dataset=train_dataset,
                num_vocab=oov_idx-1,
                num_examples=num_examples, lookup_table=tok2ids_table,
                ids2tok_dict=id2tok, ids2tok_talbe=ids2tok_table,
                sess=sess)

