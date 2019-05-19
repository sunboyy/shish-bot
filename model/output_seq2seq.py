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

with open('word_to_index.pickle', 'rb') as handle:
    word_to_index = pickle.load(handle)
    
with open('index_to_word.pickle', 'rb') as handle:
    index_to_word = pickle.load(handle)

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

batch_size = 64
num_epochs = 1000
emb_dim = 300
unk_id = 1
learning_rate = 0.001
maxLength = 10
vocab_size = len(word_to_index)
start_id = word_to_index['start_id']
end_id = word_to_index['end_id']

def initiate_model():
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    
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
    
    return sess, net_rnn, y, encode_seqs2, decode_seqs2

##### new new new ######

def get_answer_from_model(seed):
    global sess, net_rnn, y, encode_seqs2, decode_seqs2
    seed_id = [word_to_index.get(w, unk_id) for w in seed.split(" ")]

    # Encode and get state
    state = sess.run(net_rnn.final_state_encode,
                    {encode_seqs2: [seed_id]})
    # Decode, feed start_id and get first word
    o, state = sess.run([y, net_rnn.final_state_decode],
                    {net_rnn.initial_state_decode: state,
                    decode_seqs2: [[start_id]]})
    while True:
        w_id = tl.nlp.sample_top(o[0], top_k=3)
        if w_id != unk_id:
            break
    w = index_to_word[w_id]
    # Decode and feed state iteratively
    sentence = [w]
    for _ in range(maxLength): # max sentence length
        o, state = sess.run([y, net_rnn.final_state_decode],
                        {net_rnn.initial_state_decode: state,
                        decode_seqs2: [[w_id]]})
        while True:
            w_id = tl.nlp.sample_top(o[0], top_k=2)
            if w_id != unk_id:
                break
        w = index_to_word[w_id]
        if w_id == end_id:
            break
        sentence = sentence + [w]
    return sentence

######## I don't know if you change this #################
sess, net_rnn, y, encode_seqs2, decode_seqs2 = initiate_model()

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import re
import string

def clean_stopword(text):
    # Apply this code to every textual string
    word_list = text.split() 
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    text = ' '.join(filtered_words)
    return text


# # Remove Punctuatio
def remove_punc(text):
    punctuation = set(string.punctuation)
    except_punc = ['?', '!', '\"', ',', '.']
    for ex in except_punc:
        punctuation.remove(ex)
    out = []
    for e in list(text):

        if e in except_punc:
            if e != '.' :
                out.append(" "+e)
            else :
                out.append(".")
        elif e not in punctuation or e is '\'':
            out.append(e)
        else :
            out.append(" ")
    return "".join(out)

# # Split contraction (include 'em 'til)
def split_contraction(phrase):
    # convert two types of single qoute
    phrase = re.sub(r"’", "'", phrase)
    # specific
    phrase = re.sub(r"won\'t", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    
    phrase = re.sub(r"\'em ", "them ", phrase)
    phrase = re.sub(r"\'til", "until", phrase)
    
    phrase = re.sub(r"y\'", " you ", phrase)
    
    phrase = re.sub(r"in'", "ing", phrase)
    
    # general

    phrase = re.sub(r"n \'t", " n\'t", phrase)
    
    phrase = re.sub(r"n\'t", " n\'t", phrase)
    phrase = re.sub(r"\'re", " 're", phrase)
    phrase = re.sub(r"\'s", " 's", phrase)
    phrase = re.sub(r"\'d", " 'd", phrase)
    phrase = re.sub(r"\'ll", " 'll", phrase)
    phrase = re.sub(r"\'ve", " 've", phrase)
    phrase = re.sub(r"\'m", " 'm", phrase)

    EN_WHITELIST = "0123456789abcdefghijklmnopqrstuvwxyz '<>"

    tmp = phrase
    phrase = ''
    for c in tmp:
        if c in EN_WHITELIST:
            phrase += c
        else:
            phrase += ' '
    return phrase


# # Group up contigous white space
def group_space(text):
    return " ".join(text.split())

# # ' Expand the qoutation mark '
def expand_qoute(text):
    # remove punctuation first
    out = []
    for i in range(len(text)):
        if text[i] == "\'": 
            if(i!=0 and text[i-1] == ' '): #start qoute
                out.append("' ")
            elif(text[i+1] == ' ' and text[i-1]!=" "): #end qoute
                out.append(" '")
        else: out.append(text[i])
    return "".join(out)


# # Pad < end >
def find_dot(text):
    index = -1 
    reverse_text = text[::-1]
    for i in range(len(reverse_text)):
        if(reverse_text[i] =='.'):
            index = len(text)-i
        else:
            break
    return index

def pad_end_of_sentence(sentences):
    out = []
    for sentence in sentences:
        if len(sentence) > 0 and sentence[-1] == '.':
            out.append(sentence[:find_dot(sentence)-1])
        else: 
            out.append(sentence)
    return " <end> ".join(out) + ' <end>'
 
# ## Capital check
def capital_clean(sentence):
    return sentence.lower()

# ## Check symbols
def have_alphabet(sentence):
    for char in sentence:
        if char.isalpha():
            return True
    return False

# # Clean Function
def clean_data_main(line):
    sentences = sent_tokenize(line)
    out_sentences = []
    for sentence in sentences :
        if not have_alphabet(sentence):
            continue
        x = capital_clean(sentence)
        x = split_contraction(x)
        x = remove_punc(x)
        x = group_space(x)
        out_sentences.append(x)     
    sentence = pad_end_of_sentence(out_sentences)
    return sentence


# # Reverse Function
def recontraction(phrase):
    phrase = re.sub(r"’", "'", phrase)
    # specific
    phrase = re.sub(r"will not", "won\'t", phrase)
    phrase = re.sub(r"can not", "can\'t", phrase)
    
    phrase = re.sub(r"them", "\'em", phrase)
    phrase = re.sub(r"until", "\'til", phrase)
    
    # general

    phrase = re.sub(r" n\'t", "n't", phrase)
    phrase = re.sub(r" \'re", "'re", phrase)
    phrase = re.sub(r" \'s", "'s", phrase)
    phrase = re.sub(r" \'d", "'d", phrase)
    phrase = re.sub(r" \'ll", "'ll", phrase)
    phrase = re.sub(r" \'ve", "'ve", phrase)
    phrase = re.sub(r" \'’m", "'m", phrase)
    
    return phrase

import pickle
with open('lowerToCappital.pkl', 'rb') as handle:
    lowerToCapital = pickle.load(handle)

def apply_capital(line):
    out = []
    startWord = True
    for word in line.split():
        if(startWord) :
            out.append(word.capitalize())
            startWord = False
            continue
        if word in lowerToCapital:
            out.append(lowerToCapital[word])
        else :
            out.append(word)
    return " ".join(out)

def remove_space_between_punctuation(line) :
    out = ''
    punctuation = set(string.punctuation)
    for i in range(len(line)) :
        try:
            if line[i] == ' ' and line[i-1] in punctuation and line[i+1] in punctuation:
                pass
            else :
                out+=line[i]
        except :
            pass
    return out            

def reverse_clean(data) :
    first_question = ['who', 'what', 'where', 'when', 'why', 'how', 'do', 'does', 'is', 'are', 'am', 'did', 'have', 'has', 'had', 'can', 'could', 'may', 'might', 'would']
    out = []
    sentences = data.split('<end>')
    for sentence in sentences:
        if len(sentence) <= 0:
            continue
        x = sentence
        x = recontraction(sentence)
        x = remove_space_between_punctuation(x)
        x = apply_capital(x)
        x = x.strip()
        try:
            if x.split()[0].split("'")[0].lower() not in first_question:
                x += '.' 
            else:
                x += '?'
        except:
            x += '.'
        out.append(x)
    return " ".join(out)

def rule_based(seed):
    seed = seed.split()
    cntUNK, cnt = 0, 0
    for word in seed:
        if word not in word_to_index:
            cntUNK += 1
        cnt += 1
    if cnt > 10:
        return True
    if cnt > 3 and cntUNK > 1:
        return True
    elif cnt <= 3 and cntUNK:
        return True
    return False

def get_answer(seed):
    try:
        seed = clean_data_main(seed)
        if rule_based(seed):
            return "I don't know what you are talking about."
        sentence = get_answer_from_model(seed)
        sentence = ' '.join(sentence)
        sentence = reverse_clean(sentence)
    except:
        sentence = "I don't know what you are talking about"
        pass
    return sentence

seeds = ["Hi",
         "What do you do?",
         "What do you mean?",
         "I don't know what you're talking about."]
for seed in seeds:
        print("Query >", seed)
        for i in range(50):
            sentence = get_answer(seed)
            print(" >", sentence)

