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



def test_batch_iter(evaluate_every):

    processor = learn.preprocessing.VocabularyProcessor.restore(
        "../temp/vocab")
    processor.max_document_length = 800

    raw_data, raw_label = load_data("train")
    train_data = []
    train_label = []
    for rd, rl in zip(raw_data, raw_label):
        # for each task in data
        tmp_data = []
        tmp_label = []
        rd = list(processor.transform(rd))  # generator -> list
        for tmp_x, tmp_y in zip(rd, rl):
            tmp_x = tmp_x.tolist()
            if np.sum(tmp_x) != 0:
                tmp_data.append(tmp_x)
                tmp_label.append(tmp_y)
        train_data.append(tmp_data)
        train_label.append(tmp_label)
    del raw_data, raw_label

    raw_data, raw_label = load_data("test")
    test_data = []
    test_label = []
    for rd, rl in zip(raw_data, raw_label):
        # for each task in data
        tmp_data = []
        tmp_label = []
        rd = list(processor.transform(rd))  # generator -> list
        for tmp_x, tmp_y in zip(rd, rl):
            tmp_x = tmp_x.tolist()
            if np.sum(tmp_x) != 0:
                tmp_data.append(tmp_x)
                tmp_label.append(tmp_y)
        test_data.append(tmp_data)
        test_label.append(tmp_label)
    del raw_data, raw_label

    i = 0
    for task, batch in batch_iter(train_data, train_label, 100, 1, False):    

        x_train, y_train = zip(*batch)
        print(np.shape(x_train))
        print(np.shape(y_train))
        print(task)

        if i % evaluate_every == 0 and i:
            cnt = 0
            for task, batch in batch_iter(test_data, test_label, 50, 1, True):
                print("Entering Loop")
                x_dev, y_dev = zip(*batch)
                print(np.shape(x_dev))
                print(np.shape(y_dev))
                print(task)
                cnt += 1 
                if cnt == 4:
                    break

        i += 1
        pass


if __name__ == "__main__":
    test_batch_iter(5)
    pass