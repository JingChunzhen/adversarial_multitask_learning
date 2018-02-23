import os
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


def _end_batch_iter(epoch_count):
    flag = True
    for i in range(len(epoch_count)):
        if self.epoch_count[i] != epochs:
            flag = False
            break
    return flag


def batch_iter(data, batch_size, epochs, shuffle):
    """
    specific task data and label are chosen randomly, \
    each batch of data come from the same task, but different sentiment polarity
    """
    start_indices = [0] * len(params["task"])
    epoch_count = [0] * len(params["task"])
    while True:
        if _end_batch_iter():
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
