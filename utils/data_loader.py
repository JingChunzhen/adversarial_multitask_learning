import os
import random
import re

import numpy as np
import pandas as pd
import tensorflow as tf
import yaml
from tensorflow.contrib import learn

with open("../config/config.yaml", 'r') as f:
    params = yaml.load(f)


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    this is very crude
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(train_or_test):
    """
    Arg:
        train_or_test (string): can either be train or test 
    Returns:
        data (list), label (list)
    """
    data = []
    label = []
    for task in params["task"]:
        file_in = "../data/mtl-dataset/{}.task.{}".format(task, train_or_test)
        reviews = []
        polarities = []
        with open(file_in, "r", encoding='ISO-8859-1') as f:
            for line in f.readlines():
                line = clean_str(line)
                polarities.append([1, 0] if int(line[0]) == 0 else [0, 1])
                review = line[1:].strip()
                reviews.append(review)
        data.append(reviews)
        label.append(polarities)
    return data, label


def _end_batch_iter(epoch_count, epochs):
    flag = True
    for i in range(len(epoch_count)):
        if epoch_count[i] != epochs:
            flag = False
            break
    return flag


def batch_iter(corpus, label, batch_size, epochs, shuffle):
    """
    # TODO
    specific task data and label are chosen randomly, \
    each batch of data come from the same task, but different sentiment polarity
    """
    start_indices = [0] * len(params["task"])
    epoch_count = [0] * len(params["task"])
    data = []
    for i in range(len(params["task"])):
        array = np.array(list(zip(corpus[i], label[i])))
        data.append(array)
    while True:
        if _end_batch_iter(epoch_count, epochs):
            break
        else:
            while True:
                task = random.randint(0, len(params["task"]) - 1)
                if epoch_count[task] != epochs:
                    break
            data_size = len(data[task])
            start_index = start_indices[task]
            end_index = min(start_index + batch_size, data_size)
            start_indices[task] = end_index if end_index != data_size else 0
            epoch_count[task] += (0 if end_index != data_size else 1)

            if end_index == data_size and epoch_count[task] < epochs and shuffle:
                shuffled_indices = np.random.permutation(
                    np.range(data_size))  # TODO need to be tested
                data[task] = data[task][shuffled_indices]
                epoch_count[task] += 1
            elif end_index == data_size:
                epoch_count[task] += 1

            yield task, data[task][start_index: end_index]


# load data and batch_iter for transfer model

def load_data_v2(task, train_or_test):
    """
    Args:
        task (string): a target domain of transfer model 
        train_or_test (string): can either be "train" or "test"
    """
    file_in = "../data/mtl-dataset/{}.task.{}".format(task, train_or_test)
    reviews = []
    polarities = []
    with open(file_in, "r", encoding='ISO-8859-1') as f:
        for line in f.readlines():
            line = clean_str(line)
            polarities.append([1, 0] if int(line[0]) == 0 else [0, 1])
            review = line[1:].strip()
            reviews.append(review)
    
    return reviews, polarities


def batch_iter_v2(data, batch_size, epochs, shuffle=True):
    """
    Return:
        a batch size of a data 
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

# load data and batch_iter for transfer model (Twitter-Airline)


def load_data_v3():
    '''    
    load data (Twitter Airlines)in transfer trainer num_classes can only be two            
    Returns:
        x (list of string): data
        y (list of integer): labels
    '''
    x, y = [], []

    file_name = "../data/Tweets.csv"
    df = pd.read_csv(file_name)

    df = df[df["airline_sentiment"] !=
            "neutral"][["airline_sentiment", "text"]]
    df = df.dropna(axis=0, how="any")

    reviews = df['text'].tolist()
    labels = df['airline_sentiment'].tolist()
    x = [clean_str(review) for review in reviews]
    for label in labels:
        y.append([0, 1] if label == "negative" else [1, 0])

    return x, y


def batch_iter_v3(data, batch_size, num_epochs, shuffle=True):
    """
    batch iteration in transfer trainer 
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
