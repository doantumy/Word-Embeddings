import nltk, re
import numpy as np
from keras.preprocessing.sequence import pad_sequences

special_characters = re.compile("[^A-Za-z0-9 ]")

def clean_sentence(data):
    '''
    Define function to clean up the data by removing HTML tags, stopwords, extra space, non-characters
    Arguments:
        data: input data for cleaning
        
    Return:
        List of words
    '''
    data = data.lower().replace("<br />", " ")
    data = data.replace("-", " ")
    data = data.replace(".", ". ")
    data = re.sub("  ", " ", data)
    return re.sub(special_characters, "", data.lower())


def mini_batch(x, y, batch_size):
    total_review = x.shape[0]
    mini_batches = []
    number_of_batches = total_review//batch_size
    for i in range(0, number_of_batches):
        mini_x = x[(i*batch_size):(i+1)*batch_size, :]
        mini_y = y[(i*batch_size):(i+1)*batch_size]
        mini_batch = (mini_x, mini_y)
        mini_batches.append(mini_batch)
    return mini_batches

def format_train_review(train_review, max_seq_len, word_list):
    '''
    Arguments:
        word_index: words array containing index for words, its index is corresponding
                    to index in the embedding vector
        max_seq_len: this is our maximum review length that has been set to 300 in our graph, 
                     any thing longer than this will be cut.
    
    Return:
        Padding review with the shape of [max_seq_len]   
    '''
    count = 0
    review_clean = clean_sentence(train_review)
    review_split = review_clean.split()
    if len(review_split) > max_seq_len:
        review_max_len = review_split[-max_seq_len:]
    else: 
        review_max_len = review_split

    len_rev = len(review_max_len)
    temp_rvw = np.zeros(len_rev, dtype = 'int32')
    for word in review_max_len:
        try:
            temp_rvw[count] = word_list.index(word)
        except ValueError:
            temp_rvw[count] = word_list.index('unk') # if not found, values will be zero
        count += 1
    
    return temp_rvw


def format_user_review(train_review, batch_size, max_seq_len, word_list):
    '''
    Arguments:
        word_index: words array containing index for words, its index is corresponding
                    to index in the embedding vector
        max_seq_len: this is our maximum review length that has been set to 300 in our graph, 
                     any thing longer than this will be cut.
        word_index: words dictionary containing index for words, its index is corresponding
                    to index in the embedding matrix
    
    Return:
        Padding review with the shape of [batch_size, max_seq_len]
        Only the first row in this array has value, the rest is zeros as we only have 1 review in this batch.   
    '''
    
    rvw = np.zeros([batch_size, max_seq_len], dtype='int32')
    count = 0
    review_clean = clean_sentence(train_review)
    review_split = review_clean.split()
    
    if len(review_split) > max_seq_len:
        review_max_len = review_split[-max_seq_len:]
    else: 
        review_max_len = review_split

    len_rev = len(review_max_len)
    temp_rvw = np.zeros(len_rev, dtype = 'int32')
    for word in review_max_len:
        try:
            temp_rvw[count] = word_list.index(word)
        except ValueError:
            temp_rvw[count] = word_list.index('unk') # if not found, values will be zero
        count += 1
    
    review_pad = pad_sequences([temp_rvw], maxlen = max_seq_len, padding='pre')
    rvw[batch_size-1] = review_pad
    
    return rvw