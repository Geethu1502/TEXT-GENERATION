#!/usr/bin/env python
# coding: utf-8

# In[11]:


# import dependencies
import numpy
import sys
import nltk
nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

# load data
file=open('frankenstien-2.txt').read()
# In[13]:


# tokenization
# standardization
def tokenize_words(input):
    input=input.lower()
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(input)
    filtered=filter(lambda token: token not in stopwords.words('english'),tokens)
    return " ". join(filtered)  

processed_inputs=tokenize_words(file)
    


# In[15]:


# chars to numbers
chars= sorted(list(set(processed_inputs)))
char_to_num= dict((c,i) for i,c in enumerate (chars))


# In[16]:


# check if words to chars or chars to num(?!) has worked?
input_len=len(processed_inputs)
vocab_len=len(chars)
print("total number of characters:", input_len)
print(" total vocab:", vocab_len)


# In[17]:


#seq length
seq_length=100
x_data=[]
y_data=[]


# In[19]:


# loop through the sequence
for i in range(0,input_len-seq_length,1):
    in_seq=processed_inputs[i:i+seq_length]
    out_seq=processed_inputs[i+seq_length]
    x_data.append([char_to_num[char] for char in in_seq])
    y_data.append(char_to_num[out_seq])

n_patterns=len(x_data)
print("total patterns:",n_patterns)


# In[21]:


# convert input sequence to np array and so on
x=numpy.reshape(x_data, (n_patterns,seq_length, 1))
x=x/float(vocab_len)


# In[23]:


# one_hot encoding
y=np_utils.to_categorical(y_data)


# In[29]:


# creating the model
model=Sequential()
model.add(LSTM(256,input_shape=(x.shape[1], x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256,return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1],activation='softmax'))


# In[30]:


# compile the model
model.compile(loss='categorical_crossentropy',optimizer='adam')


# In[31]:


# saving weights
filepath=' model_weights_saved_hdf5'
checkpoint=ModelCheckpoint(filepath,monitor='loss', verbose=1,save_best_only=True,mode='min')
desired_callbacks=[checkpoint]


# In[ ]:


# fit model and let it train
model.fit(x,y, epochs=4, batch_size=256, callbacks=desired_callbacks)


# In[ ]:


# recombile the model with the saved weights
file name=' model_weights_saved.hdf5'
model.load_weights(filename)
model.compile(loss='categorical_crossentrophy',optimizer='adam')


# In[ ]:


# output of the model back into the characters
num_to_char=dict((i,c) for i,c in enumerate(chars))


# In[ ]:


# random seed to help generate
start=numpy.random.randit(0, len(x_data) -1)
pattern=x_data[start]
print(" Random seed:")
print("\"",''.join([num_to_char[value] for value in pattern]), "\"")


# In[ ]:




