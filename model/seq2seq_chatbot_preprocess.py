
# coding: utf-8

# In[6]:


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


# In[3]:


EN_WHITELIST = "0123456789abcdefghijklmnopqrstuvwxyz '<>" # space is included in whitelist
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\'.'

def clean_data(line):
    res = []
    line = line.strip('""')
    line = line.replace('�', ' ')
    line = line.replace('\x92', "'")
    line = line.replace("n't", " n't")
    line = line.replace("n 't", " n't")
    line = line.replace("i'm", "i 'm")
    line = line.replace('\x91', "")
    line = line.replace('\x97', "")
    line = line.replace('\x93', "")
    
    tmp = line
    line = ''
    for c in tmp:
        if c in EN_WHITELIST:
            line += c
        else:
            line += ' '
    
    line = line.split()
    
    for word in line:
        if word != '':
            res += [word]
            
    return res

load = open('data/cornell.csv', 'rb')
lines = load.readlines()
lines = lines[1:]

for i in range(len(lines)):
    lines[i] = lines[i].decode('utf-8').strip()
    
trainX, trainY, tmp = list(), list(), list()

characters = set()    
data = []

chk = False
for i in range(len(lines)):
    lines[i] = lines[i].split(',', 2)
    
    if lines[i][0] != 'Scene':
        tmpData = clean_data(lines[i][2])
        if len(tmpData) <= 10:
            data += [tmpData]
        if not chk and len(tmp) <= 10 and len(tmpData) <= 10:
            trainX += [tmp]
            trainY += [tmpData]
        chk = False
        tmp = tmpData
    else:
        chk = True
        
    characters.add(lines[i][0])

cnt_word = {}
sm = 0

for line in data:
    line = line
    for word in line:
        if word not in cnt_word:
            cnt_word[word] = 1
            sm += 1
        else:
            if cnt_word[word] == 1:
                sm -= 1
            cnt_word[word] += 1


word_to_index = {}
index_to_word = ['_', '<unk>']

word_to_index['_'] = 0
word_to_index['<unk>'] = 1

i = 2
for word in cnt_word:
    if cnt_word[word] > 1 and word not in word_to_index:
        index_to_word.append(word)
        word_to_index[word] = i
        i += 1
        
start_id = len(word_to_index)
end_id = len(word_to_index) + 1
        
word_to_index.update({'start_id': start_id})
word_to_index.update({'end_id': end_id})
index_to_word = index_to_word + ['start_id', 'end_id']

def replace_unk(line):
    res = []
    for word in line:
        if word in word_to_index:
            res += [word]
        else:
            res += ['<unk>']
    return res

for i in range(len(trainX)):
    trainX[i] = replace_unk(trainX[i])
    trainY[i] = replace_unk(trainY[i])

total_word_count = len(index_to_word)

maxLength = 10

vocab_size = len(word_to_index)


# In[4]:


from sklearn.model_selection import train_test_split
train_x, test_x = train_test_split(trainX, test_size=0.1, random_state=57)
train_y, test_y = train_test_split(trainY, test_size=0.1, random_state=57)

train_x = np.array(train_x)
train_y = np.array(train_y)


# In[7]:


with open('word_to_index.pickle', 'wb') as handle:
    pickle.dump(word_to_index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('index_to_word.pickle', 'wb') as handle:
    pickle.dump(index_to_word, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[8]:


with open('trainX.pickle', 'wb') as handle:
    pickle.dump(train_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('trainY.pickle', 'wb') as handle:
    pickle.dump(train_y, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('testX.pickle', 'wb') as handle:
    pickle.dump(test_x, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
with open('testY.pickle', 'wb') as handle:
    pickle.dump(test_y, handle, protocol=pickle.HIGHEST_PROTOCOL)

