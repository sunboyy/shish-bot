
# coding: utf-8

# In[1]:


from pickle import load
import pickle
import numpy as np
import time
import click
import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tqdm import tqdm
from sklearn.utils import shuffle
from tensorlayer.layers import DenseLayer, EmbeddingInputlayer, Seq2Seq, retrieve_seq_length_op2
import random


# In[2]:


sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)


# In[4]:


with open('word_to_index.pickle', 'rb') as handle:
    word_to_index = pickle.load(handle)
    
with open('index_to_word.pickle', 'rb') as handle:
    index_to_word = pickle.load(handle)
    
with open('trainX.pickle', 'rb') as handle:
    train_x = pickle.load(handle)
    
with open('trainY.pickle', 'rb') as handle:
    train_y = pickle.load(handle)
    
with open('testX.pickle', 'rb') as handle:
    test_x = pickle.load(handle)
    
with open('testY.pickle', 'rb') as handle:
    test_y = pickle.load(handle)


# In[5]:


batch_size = 64


# In[6]:


def create_model(encode_seqs, decode_seqs, src_vocab_size, emb_dim, is_train=True, reuse=False):
    with tf.variable_scope("model", reuse=reuse):
        
        with tf.variable_scope("embedding") as vs:
            net_encode = EmbeddingInputlayer(
                inputs = encode_seqs,
                vocabulary_size = src_vocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
            vs.reuse_variables()
            net_decode = EmbeddingInputlayer(
                inputs = decode_seqs,
                vocabulary_size = src_vocab_size,
                embedding_size = emb_dim,
                name = 'seq_embedding')
            
        net_rnn = Seq2Seq(net_encode, net_decode,
                cell_fn = tf.nn.rnn_cell.LSTMCell,
                n_hidden = emb_dim,
                initializer = tf.random_uniform_initializer(-0.1, 0.1),
                encode_sequence_length = retrieve_seq_length_op2(encode_seqs),
                decode_sequence_length = retrieve_seq_length_op2(decode_seqs),
                initial_state_encode = None,
                dropout = (0.5 if is_train else None),
                n_layer = 3,
                return_seq_2d = True,
                name = 'seq2seq')

        net_out = DenseLayer(net_rnn, n_units=src_vocab_size, act=tf.identity, name='output')
    return net_out, net_rnn


# In[7]:


def encode_input(trainX):
    seq = []
    for i in range(len(trainX)):
        sequence = trainX[i]
        tmpSeq = []
        for i in range(min(maxLength, len(sequence))):
            tmpSeq = tmpSeq + [word_to_index[sequence[i]]]
        seq += [tmpSeq]
    X = seq
    
    return X

def decode_input(trainY):
    #for output
    seq = []

    for i in range(len(trainY)):
        sequence = trainY[i]
        tmpSeq = []
        for i in range(min(maxLength, len(sequence))):
            tmpSeq = tmpSeq + [word_to_index[sequence[i]]]
        seq += [tmpSeq]
#     encoded = to_categorical(seq, num_classes = vocab_size)
    Y = seq
    return Y


# In[18]:


num_epochs = 1000
n_step = len(train_x) // batch_size
emb_dim = 300
unk_id = 1
learning_rate = 0.001
maxLength = 10
vocab_size = len(word_to_index)
start_id = word_to_index['start_id']
end_id = word_to_index['end_id']

train_data = encode_input(train_x)
train_label = decode_input(train_y)


# In[16]:


# Init Session
tf.reset_default_graph()
sess = tf.Session(config=sess_config)

# Training Data Placeholders
encode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="encode_seqs")
decode_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="decode_seqs")
target_seqs = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_seqs")
target_mask = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name="target_mask") 

net_out, _ = create_model(encode_seqs, decode_seqs, vocab_size, emb_dim, is_train=True, reuse=False)
net_out.print_params(False)

# Inference Data Placeholders
encode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="encode_seqs")
decode_seqs2 = tf.placeholder(dtype=tf.int64, shape=[1, None], name="decode_seqs")

net, net_rnn = create_model(encode_seqs2, decode_seqs2, vocab_size, emb_dim, is_train=False, reuse=True)
y = tf.nn.softmax(net.outputs)

# Loss Function
loss = tl.cost.cross_entropy_seq_with_mask(logits=net_out.outputs, target_seqs=target_seqs, 
                                            input_mask=target_mask, return_details=False, name='cost')

# Optimizer
train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Init Vars
sess.run(tf.global_variables_initializer())

# Load Model
tl.files.load_and_assign_npz(sess=sess, name='model.npz', network=net)


# In[17]:


def inference(seed):
    seed_id = [word_to_index.get(w, unk_id) for w in seed.split(" ")]

    # Encode and get state
    state = sess.run(net_rnn.final_state_encode,
                    {encode_seqs2: [seed_id]})
    # Decode, feed start_id and get first word [https://github.com/zsdonghao/tensorlayer/blob/master/example/tutorial_ptb_lstm_state_is_tuple.py]
    o, state = sess.run([y, net_rnn.final_state_decode],
                    {net_rnn.initial_state_decode: state,
                    decode_seqs2: [[start_id]]})
    w_id = tl.nlp.sample_top(o[0], top_k=3)
    w = index_to_word[w_id]
    # Decode and feed state iteratively
    sentence = [w]
    for _ in range(maxLength): # max sentence length
        o, state = sess.run([y, net_rnn.final_state_decode],
                        {net_rnn.initial_state_decode: state,
                        decode_seqs2: [[w_id]]})
        w_id = tl.nlp.sample_top(o[0], top_k=2)
        w = index_to_word[w_id]
        if w_id == end_id:
            break
        sentence = sentence + [w]
    return sentence

seeds = ["are you okay <end>",
         "what do you do <end>",
         "what do you mean <end>"]
for seed in seeds:
        print("Query >", seed)
        for _ in range(5):
            sentence = inference(seed)
            print(" >", ' '.join(sentence))

seeds = ["are you okay <end>",
         "what do you do <end>",
         "what do you mean <end>"]
for epoch in range(num_epochs):
    trainX, trainY = shuffle(train_data, train_label, random_state=0)
    total_loss, n_iter = 0, 0
    for X, Y in tqdm(tl.iterate.minibatches(inputs=trainX, targets=trainY, batch_size=batch_size, shuffle=False), 
                    total=n_step, desc='Epoch[{}/{}]'.format(epoch + 1, num_epochs), leave=False):

        X = tl.prepro.pad_sequences(X)
        _target_seqs = tl.prepro.sequences_add_end_id(Y, end_id=end_id)
        _target_seqs = tl.prepro.pad_sequences(_target_seqs)
        _decode_seqs = tl.prepro.sequences_add_start_id(Y, start_id=start_id, remove_last=False)
        _decode_seqs = tl.prepro.pad_sequences(_decode_seqs)
        _target_mask = tl.prepro.sequences_get_mask(_target_seqs)

        _, loss_iter = sess.run([train_op, loss], {encode_seqs: X, decode_seqs: _decode_seqs,
                        target_seqs: _target_seqs, target_mask: _target_mask})
        total_loss += loss_iter
        n_iter += 1

    # printing average loss after every epoch
    print('Epoch [{}/{}]: loss {:.4f}'.format(epoch + 1, num_epochs, total_loss / n_iter))

    # inference after every epoch
    for seed in seeds:
        print("Query >", seed)
        for _ in range(5):
            sentence = inference(seed)
            print(" >", ' '.join(sentence))

    # saving the model
    tl.files.save_npz(net.all_params, name='model.npz', sess=sess)

sess.close()

