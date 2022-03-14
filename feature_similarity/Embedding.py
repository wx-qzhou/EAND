import re
import torch
import numpy as np
from numpy import dot
from numpy.linalg import norm
from utils import args
from utils import path_file
from os.path import join
from scipy.stats import pearsonr
from gensim.models.word2vec import Word2Vec

'''norm'''
def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    return (data - mu) / sigma

'''the embedding of word2vec'''
def word_2_vec_embedding(sentence):
    model = Word2Vec(sentence, min_count=1, size=args.size, workers=args.worker, sg=1)
    model.save(join(path_file.preprocessed_path, '{}.emb'.format("features")))

def word_2_vec(sentence, model, idf):
    vectors = []
    sum_weight = 0
    for token in sentence[0]:
        if not token in model.wv:
            continue
        weight = 1
        if idf and token in idf:
            weight = idf[token]
        v = model.wv[token] * weight
        vectors.append(v)
        sum_weight += weight
    if len(vectors) == 0:
        print('all tokens not in w2v models')
        return None
    emb = np.sum(vectors, axis=0)
    emb /= sum_weight
    return emb

def load():
    return Word2Vec.load(join(path_file.preprocessed_path, 'features.emb'))

def embedding_with_word2vec(sentence1, sentence2, model, idf):
    sentence1 = [sentence1]
    sentence2 = [sentence2]
    emb1 = standardization(word_2_vec(sentence1, model, idf))
    emb2 = standardization(word_2_vec(sentence2, model, idf))
    return emb1, emb2

"""cos similarity"""
def cosine_sim(v1, v2):
    cos_sim = dot(v1, v2) / (norm(v1) * norm(v2))
    return cos_sim

"""pearsonr similarity"""
def pearsonr_(v1, v2):
    return pearsonr(v1, v2)[0]

# if __name__ == "__main__":
#     string = [
#         "__TITLE__new",
#         "__TITLE__method",
#         "__TITLE__solv",
#         "__TITLE__drop",
#         "__TITLE__call",
#         "__TITLE__cdma",
#         "__TITLE__cellular",
#         "__TITLE__system",
#         "__VENUE__ieee",
#         "__VENUE__vehicular",
#         "__VENUE__technology",
#         "__VENUE__conference",
#         "__ABSTRACT__mobil",
#         "__ABSTRACT__environ",
#         "__ABSTRACT__drop",
#         "__ABSTRACT__call",
#         "__ABSTRACT__result",
#         "__ABSTRACT__shadow",
#         "__ABSTRACT__rapid",
#         "__ABSTRACT__signal",
#         "__ABSTRACT__loss",
#         "__ABSTRACT__paper",
#         "__ABSTRACT__present",
#         "__ABSTRACT__implement",
#         "__ABSTRACT__transpar",
#         "__ABSTRACT__reconnect",
#         "__ABSTRACT__procedur",
#         "__ABSTRACT__trp",
#         "__ABSTRACT__effici",
#         "__ABSTRACT__algorithm",
#         "__ABSTRACT__adapt",
#         "__ABSTRACT__easili",
#         "__ABSTRACT__benefit",
#         "__ABSTRACT__decreas",
#         "__ABSTRACT__drop",
#         "__ABSTRACT__call",
#         "__ABSTRACT__simul",
#         "__ABSTRACT__typic",
#         "__ABSTRACT__cellular",
#         "__ABSTRACT__system",
#         "__ABSTRACT__fewer",
#         "__ABSTRACT__drop",
#         "__ABSTRACT__call",
#         "__ABSTRACT__trp",
#         "__ABSTRACT__compar",
#         "__ABSTRACT__convent",
#         "__ABSTRACT__procedur",
#         "__ABSTRACT__benefit",
#         "__ABSTRACT__come",
#         "__ABSTRACT__expens",
#         "__ABSTRACT__slight",
#         "__ABSTRACT__increas",
#         "__ABSTRACT__1",
#         "__ABSTRACT__block",
#         "__ABSTRACT__call",
#         "__ABSTRACT__percentag"
#     ]
#     string1 = string
#     model = load()
#     emb1, emb2 = embedding_with_word2vec(string, string1, model, {})
#     print(cosine_sim( emb1, emb2 ))
#     # v1, v2 = embedding(string, string1, model)
#     # print(cosine_sim(v1, v2))
    
#     print(pearsonr_(emb1, emb2))