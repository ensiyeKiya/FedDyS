import re
import numpy as np
import torch


ALL_LETTERS = "\n !\"&'(),-.0123456789:;>?ABCDEFGHIJKLMNOPQRSTUVWXYZ[]abcdefghijklmnopqrstuvwxyz}"
NUM_LETTERS = len(ALL_LETTERS)
MAX_LEN = 1000

def _one_hot(index, size):
    '''returns one-hot vector with given size and value 1 at given index
    '''
    vec = [0 for _ in range(size)]
    vec[int(index)] = 1
    return vec


def letter_to_vec(letter):
    '''returns one-hot representation of given letter
    '''
    index = ALL_LETTERS.find(letter)
    return _one_hot(index, NUM_LETTERS)


def word_to_indices(word):
    '''returns a list of character indices
    Args:
        word: string

    Return:
        indices: int list with length len(word)
    '''
    indices = []
    for c in word:
        indices.append(max(ALL_LETTERS.find(c), 0))
    return indices


# ------------------------
# utils for sent140 dataset


def split_line(line):
    '''split given line/phrase into list of words
    Args:
        line: string representing phrase to be split

    Return:
        list of strings, with each string representing a word
    '''
    return re.findall(r"[\w']+|[.,!?;]", line)


def _word_to_index(word, indd):
    '''returns index of given word based on given lookup dictionary
    returns the length of the lookup dictionary if word not found
    Args:
        word: string
        indd: dictionary with string words as keys and int indices as values
    '''
    if word in indd:
        return indd[word]
    else:
        return len(indd)


def line_to_indices(line, word2id, max_words=25):
    '''converts given phrase into list of word indices

    if the phrase has more than max_words words, returns a list containing
    indices of the first max_words words
    if the phrase has less than max_words words, repeatedly appends integer
    representing unknown index to returned list until the list's length is
    max_words
    Args:
        line: string representing phrase/sequence of words
        word2id: dictionary with string words as keys and int indices as values
        max_words: maximum number of word indices in returned list
    Return:
        indl: list of word indices, one index for each word in phrase
    '''
    unk_id = len(word2id)
    line_list = split_line(line) # split phrase in words
    indl = [word2id[w] if w in word2id else unk_id for w in line_list[:max_words]]
    indl += [unk_id]*(max_words-len(indl))
    return indl


def bag_of_words(line, vocab):
    '''returns bag of words representation of given phrase using given vocab
    Args:
        line: string representing phrase to be parsed
        vocab: dictionary with words as keys and indices as values
    Return:
        integer list
    '''
    bag = [0]*len(vocab)
    words = split_line(line)
    for w in words:
        if w in vocab:
            bag[vocab[w]] += 1
    return bag


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)



def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

def process_x(raw_x_batch):
    x_batch = [word_to_indices(word) for word in raw_x_batch]
    x_batch = padding_(x_batch, MAX_LEN)
    # x_batch = np.array(x_batch).T
    return torch.tensor(x_batch)


def process_y(raw_y_batch):
    y_batch = [letter_to_vec(c) for c in raw_y_batch]
    return np.array(y_batch)
