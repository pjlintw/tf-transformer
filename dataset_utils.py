import functools
from pathlib import Path

import tensorflow as tf

import numpy as np
import random

def ftgt(tgt_name):
    """Return target file repository"""
    return f'./data/tgt.{tgt_name}.example.txt'


def fsrc(src_name):
    """Return source file repository"""
    return f'./data/src.{src_name}.example.txt'


def parser_fn(line_src, line_tgt, tok2id):
    
    # preprocess
    line_src = line_src.replace(' ', '')
    line_tgt = line_tgt.replace(' ', '')

    source = [char for char in line_src.strip()]
    target = [char for char in line_tgt.strip()]

    r = lambda x: tok2id[x] if x in tok2id else tok2id['<unk>']
    source = list(map(r, source))
    target = list(map(r, target))
    target = [tok2id['<sos>']] + target + [tok2id['<eos>']]

    tar_inp = target[:-1]
    tar_real = target[1:]

    sorce_len = len(source)

    return (source, sorce_len), (tar_inp, tar_real)


def generator_fn(src_fn, tgt_fn, tok2id):
    """Yield pased sources and targets by processing exampels from file"""
    
    with Path(fsrc(src_fn)).open(encoding='utf-8') as f_source, Path(ftgt(tgt_fn)).open(encoding='utf-8') as f_target:
        for line_src, line_tgt in zip(f_source, f_target):
            yield parser_fn(line_src, line_tgt, tok2id)


def input_fn(src_fn, tgt_fn, tok2id, num_epoch=1, batch_size=32, shuffle=False):
    """
    
        shuffle --> batch --> repeat
    """

    num_train_examples = sum([1 for _ in Path(fsrc(src_fn)).open(encoding='utf-8') ])

    shapes = (([None], ()), 
              ([None], [None]))
    types = ((tf.int32, tf.int32),
             (tf.int32, tf.int32))
    padding_values = ((0, 0),
                      (0, 0))

    data = tf.data.Dataset.from_generator(
        functools.partial(generator_fn, src_fn, tgt_fn, tok2id),
        output_shapes=shapes, output_types=types)

    if shuffle:
        data = data.shuffle(buffer_size=50)
    
    data = data.padded_batch(batch_size=batch_size, padded_shapes=shapes, padding_values=padding_values).prefetch(buffer_size=20).repeat(count=num_epoch)

    return data, num_train_examples


def batch_generator(sess, dataset):
    """Generate batch data from tensorflow dataset
    """
    one_batch = dataset.make_one_shot_iterator().get_next()

    while True:
        try:
            yield sess.run(one_batch)

        # stop if reaching end of `Dataset`
        except tf.errors.OutOfRangeError:
            print('End of dataset')
            break


def convert_tensor_to_string(x, lookup_table, sess):
    """Convert id tensor into string"""

    #bytes_arr = sess.run(x)
    return [ byte_tok.decode('utf-8') for byte_tok in bytes_arr for bytes_arr in x]



def create_tok2id_from_vocab_file(vocab_file):

    tok2id = dict()
    id2tok = dict()

    ids = 0

    with Path(vocab_file).open(encoding='utf-8') as f:
        data = f.readlines()

        # index 0 for <pad>
        if '<pad>' not in data:
            tok2id['<pad>'] = ids
            id2tok[ids] = '<pad>'

            ids += 1

        for char in data:
            char = char.strip()
            if char != '' and char != ' ' and char !='\t':
                tok2id[char] = ids
                id2tok[ids] = char
                ids += 1

    for tok in ['<sos>', '<eos>']:
        if tok not in tok2id:
            tok2id[tok] = ids
            id2tok[ids] = tok
            ids += 1

    if '<unk>' not in data:
            tok2id['<unk>'] = ids
            id2tok[ids] = '<unk>'
            ids += 1

    return tok2id, id2tok


def create_tf_hash_table(keys, values, key_dtype, value_dtype, default_value):
    """Create tensorflow lookup table"""
    return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys=keys,
                                                        values=values,
                                                        key_dtype=key_dtype,
                                                        value_dtype=value_dtype), default_value)


def convert_idx_to_token_tensor(x, idx2tok):
    """Convert int32 tensor to string tensor. 

    Args:
      x: 2D int32 tensor
      id2tok: dictionary, map index into token

    Returns:
      2D string tensor

    """
    def py_convert_idx_to_token_fn(x):
        return [ " ".join([n for batch in x for char in batch] )  ]
    return tf.py_func(py_convert_idx_to_token_fn, [x], tf.string)
