import numpy as np

import tensorflow as tf
from tensorflow.keras.datasets import imdb

from collections import Counter

from nn import dense, attention
from transformer import Transformer

import re

def batch_iter(x_data, y_data, batch_size=5, num_epochs=1, shuffle=True):

    data_size = len(x_data)
    num_batches_per_epoch = int( (data_size -1) / batch_size) + 1

    for epoch in range(num_epochs):
        print('Epoch:', epoch)

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffle_data = x_data[shuffle_indices]
            shuffle_labels = y_data[shuffle_indices]
        else:
            shuffle_data = x_data
            shuffle_labels = y_data

        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)

            if shuffle_data[start_idx:end_idx].shape[0] != 32:
                continue
            else:
                yield shuffle_data[start_idx:end_idx], shuffle_labels[start_idx:end_idx]


# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)

# keep the top 10,000 most frequently occurring words
# train_data: (25000, )   , each data is list of integers
# train_labels: (25000, ) , each label is an integer value 0 or 1
# test_data: (25000, )
# test_labels: (25000, )
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=20000)

# restore np.load for future normal use
np.load = np_load_old

# get word index
word_ids = imdb.get_word_index()
#print(word_ids)

# number of voab
c = Counter()
for data in train_data:
    c.update(data)
    

# 先算出來
n_vocab = len(c)

print(n_vocab)

# padding
x_train = tf.keras.preprocessing.sequence.pad_sequences(train_data, maxlen=80, padding='post')
x_test =  tf.keras.preprocessing.sequence.pad_sequences(test_data, maxlen=80, padding='post')

### in progressing ###
def create_input_mask(input_tensor):
    return tf.cast(input_tensor>0, tf.int32)


def imdb_y_input_fn(imdb_y, batch_size, num_classes):
    feed_y = np.zeros((batch_size, num_classes))

    for idx in range(batch_size):
        batch_idx = idx
        label_idx = imdb_y[idx]

        feed_y[batch_idx][label_idx] = 1

    return feed_y


def imdb_binary_y_input_fn(imdb_y, batch_size, num_classes):
    feed_y = np.zeros([batch_size, num_classes])

    for i in range(batch_size):
        label = imdb_y[i]

        if label == 1:
            feed_y[i][0] = 1
            
        else:
            feed_y[i][0] = 0
    return feed_y


model = Transformer(embedding_dim=128, vocab_size=n_vocab, sequence_maxlen=80, 
                    num_classes=2, num_layers=2, num_head=8, head_size=16, lr=1e-3)


with tf.Session() as sess:
  
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    total_batch = 0

    # iterate epoch
    for batch in batch_iter(x_train, train_labels, batch_size=32, num_epochs=5):
        # x shape: (batch_size, maxlen, dim)
        x, y = batch
        #print(x.shape)
        #print('x', x)
        #print(y)
        
        #input_mask = create_input_mask(x)
        # [2,2,0,0,0,0,0] with size 32

        def get_sequence_length_from_two_dim_batch(x):
            sequence_length = list()
        
            for batch_arr in x:
                seq_len = sum(batch_arr != 0)
                sequence_length.append(seq_len)

            return sequence_length
        
        seq_len = get_sequence_length_from_two_dim_batch(x)

        bs_y_data = train_labels[:10]
        
        if total_batch % 100 == 0 or total_batch == 0:   
            logits, loss, acc = sess.run([model.logits, model.loss, model.acc], feed_dict={model.inputs: x,
                                                                                           model.y_cate: y,  model.seq_len: seq_len})            
            #pred = tf.equal(tf.round(self.logits), self.y_cate)

            # batch_size of correct predictions    
            train_msg = f"Train Loss: {loss:>6.2}, Train Acc: {acc:>7.2%}"
            print(train_msg)
            # print(logits)

        _ = sess.run([model.optimizer], feed_dict={model.inputs: x, model.y_cate: y, model.seq_len: seq_len})
        total_batch += 1

        ### Iterate mask_op ###
        # print('sequence length:', seq_len)
        # batch_flag = 1
        # for aw in attn_weight:
        #     print('batch: ', batch_flag)
        #     for h in aw:
        #         print('head one')
        #         src_word = 1
        #         for src_attn in h:
        #             print(str(src_word))
        #             print(src_attn)
        #             src_word += 1
        #         break
        #     batch_flag += 1
        #     print()
        
        # break
        ### Iterate mask_op ###
