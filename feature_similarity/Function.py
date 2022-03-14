import math
import numpy as np
from numpy import dot
from utils import data_utils
from numpy.linalg import norm
from collections import Counter
from utils import path_file

# calculate tf
def tf(word, count): 
    return count[word] / sum(count.values())

# read global idf value
def idf(): 
    try:
        IDF = data_utils.load_data(path_file.feature_path, "feature_idf.pkl")
    except Exception as ex:
        print(ex)
        IDF = {}
    return IDF

# calculate the value of tf-idf
def tfidf(word, count, THREAD_VALUE, IDF):
    if not word in IDF:
        return math.log(tf(word, count) + 1) * np.mean(list(IDF.values())) / (2 * THREAD_VALUE)
    else:
        return math.log(tf(word, count) + 1) * IDF.get(word, THREAD_VALUE) / (2 * THREAD_VALUE)

def Cal_one_paper_tf_idf(word_list, THREAD_VALUE, IDF=None):
    if IDF == None:
        IDF = idf()
    count = Counter(word_list)
    tf_idf = {}
    for word in count:
        tf_idf[word] = tfidf(word, count, THREAD_VALUE, IDF)
    return tf_idf