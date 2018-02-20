import re
import os
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


def statistic_task_sequence_length():
    """
    find optimum sequence length for each task 
    """

    for task in params["task"]:
        file_in = "../data/mtl-dataset/" + task + ".task.train"
        lengths = []
        review_nums = {}
        total_num = 0
        # total_num indicate how many users

        with open(file_in, "r", encoding='ISO-8859-1') as f:
            for line in f.readlines():
                line = clean_str(line)
                label = line[0]
                review = line[1:].strip()
                length = len(review.split(" "))
                if length not in review_nums:
                    review_nums[length] = 1
                else:
                    review_nums[length] += 1
                total_num += 1

        sorted(review_nums.items(), key=lambda d: d[0])

        c, x, y = 0, [], []
        for k, v in review_nums.items():
            c += v
            percentage = c * 1.0 / total_num
            x.append(k)
            y.append(percentage)

        from pyecharts import Line
        line = Line("")

        line.add("{}评论长度百分比".format(task), x, y)
        line.render("../doc/{}.html".format(task))
        print("{} complete!".format(task))


def vocabulary_generator():
    """
    map words to ids 
    # TODO:add new words to a existed vocabulary need to be tested 
    """
    vocabulary = None
    for task, sequence_length in zip(params["task"], params["optimum_length"]):
        file_in = "../data/mtl-dataset/" + task + ".task.train"
        # TODO
        reviews = []
        with open(file_in, 'r', encoding="ISO-8859-1") as f:
            for line in f.readlines():
                line = line[1:].strip()
                review = clean_str(line).split(' ')
                reviews.append(review)

        processor = learn.preprocessing.VocabularyProcessor(
            max_document_length=sequence_length, vocabulary=vocabulary)
        processor.fit(reviews)
        vocabulary = processor.vocabulary_

    processor.save("../temp/vocab")


def convert_to_csv():
    """
    convert the data into csv format
    """
    data = []
    for suffix in ["train", "test"]:
        for task in params["task"]:
            file_in = "../data/mtl-dataset/{}.task.{}".format(task, suffix)
            with open(file_in, 'r', encoding="ISO-8859-1") as f:
                for line in f.readlines():
                    domain = task
                    label = line[0]
                    review = line[1:].strip()
                    data.append([review, domain, label])
    df = pd.DataFrame(data=data, columns=["review", "domain", "label"])
    df.to_csv("../temp/data.csv")


def load_data():
    """
    Returns:
        reviews (list of list): text
        domain (list of int): the domain of the text
        label (list of int): sentiment polarity of the text
    """    
    file_in = "../temp/data.csv"
    if not os.path.exists(file_in):
        convert_to_csv()      
    tasks = params["task"]
    df = pd.read_csv(file_in)
    reviews = [clean_str(r).split(" ") for r in df["review"].tolist()]
    label = df["label"].tolist()
    domain = [tasks.index(d) for d in df["domain"].tolist()]
    return reviews, domain, label


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """    
    Generates a batch iterator for a dataset.
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


if __name__ == "__main__":
    conver_to_csv()
    pass
