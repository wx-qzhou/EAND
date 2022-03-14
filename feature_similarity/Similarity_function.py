import math
import numpy as np
from utils.string_utils import clean_sentence
from feature_similarity.Similiarity import feature_IDF, feature_jac, feature_TF_IDF, \
    feature_dice
from feature_similarity.Embedding import cosine_sim, embedding_with_word2vec, \
    pearsonr_

def is_less_zero(temp):
    if temp < 0:
        return 0.0
    else:
        return temp

"""coauthor similarity"""
def coauthor_sim(coauthor1, coauthor2, idf):
    if coauthor1 is None or coauthor2 is None or len(coauthor1) == 0 or len(coauthor2) == 0:
        return np.nan
    similarity = 0.0
    similarity += feature_jac(coauthor1, coauthor2) # 0.5
    similarity += feature_dice(coauthor1, coauthor2) # 1.0
    similarity /= 2
    similarity += feature_TF_IDF(coauthor1, coauthor2, idf) #
    similarity += feature_IDF(coauthor1, coauthor2, idf) # 1.006
    return similarity

"""affiliation similarity"""   
def affiliation_sim(affiliation1, affiliation2, idf):
    if affiliation1 is None or affiliation2 is None or len(affiliation1) == 0 or len(affiliation2) == 0:
        return np.nan
    similarity = 0.0
    similarity += feature_jac(affiliation1, affiliation2)
    similarity += feature_dice(affiliation1, affiliation2) # 1.0
    similarity /= 2
    similarity += feature_IDF(affiliation1, affiliation2, idf)
    similarity += feature_TF_IDF(affiliation1, affiliation2, idf)
    return similarity

"""venue similarity"""    
def venue_sim(venue1, venue2, model, idf):
    similarity = 0.0
    if venue1 is None or venue2 is None or len(venue1) == 0 or len(venue2) == 0 \
        or venue1 == '' or venue2 == '':
        return np.nan
    else:
        similarity += feature_IDF(venue1, venue2, idf)
        similarity += feature_jac(venue1, venue2)
        return similarity

"""keywords"""
def keywords_sim(keywords1, keywords2, idf):
    if keywords1 is None or keywords2 is None or len(keywords1) == 0 or len(keywords2) == 0:
        return np.nan
    similarity = 0.0
    similarity += feature_IDF(keywords1, keywords2, idf)
    similarity += feature_jac(keywords1, keywords2)
    return similarity

"""title"""
def title_abstract_sim(tiltle_abstract1, tiltle_abstract2, model, idf):
    similarity = 0.0
    if tiltle_abstract1 is None or tiltle_abstract2 is None or \
        len(tiltle_abstract1) == 0 or len(tiltle_abstract2) == 0 or \
            tiltle_abstract1 == '' or tiltle_abstract2 == '':
        return np.nan
    else:
        emb1, emb2 = embedding_with_word2vec(tiltle_abstract1, tiltle_abstract2, model, idf)
        similarity += pearsonr_(emb1, emb2)
        similarity += cosine_sim(emb1, emb2)
        similarity /= 2
        similarity += feature_IDF(tiltle_abstract1, tiltle_abstract2, idf)
        return is_less_zero(similarity)

"""all_feature"""
def all_feature_sim(all_feature1, all_feature2, idf):
    similarity = 0.0
    # similarity += feature_jac(all_feature1, all_feature2)
    # similarity += feature_dice(all_feature1, all_feature2) # 1.0
    # similarity /= 2
    similarity += feature_IDF(all_feature1, all_feature2, idf)
    # similarity += feature_TF_IDF(all_feature1, all_feature2, idf)
    return similarity

"""year"""
def cal_year_sim(year_1, year_2):
    if year_1 is None or year_2 is None or year_1 == 0 or year_2 == 0 or year_1 == "" or year_2 == "":
        return np.nan
    return 1 / (1 + abs(year_1 - year_2))

