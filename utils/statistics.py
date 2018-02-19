import re

import pandas as pd
import yaml

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

import tensorflow as tf
from tensorflow.contrib import learn

def get_processor():
    """
    # TODO add new words to a existed vocabulary 
    """
    processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    processor.fit()    


if __name__ == "__main__":
    statistic_task_sequence_length()
    pass
