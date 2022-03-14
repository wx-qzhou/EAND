import multiprocessing
import pickle
from copy import copy
import numpy as np
from utils import path_file, string_utils
from feature_similarity import Similarity_function
from feature_similarity.Embedding import load

# generate the feature like : __name__feature 
def transform_feature(data, f_name):
    assert type(data) is list

    features = []
    for d in data:
        features.append("__%s__%s" % (f_name.upper(), d))

    return features

# generate the feature of coauthor
def gen_feature_author(name_list, reference_index):
    if reference_index >= 0:
        try:
            name_list.pop(reference_index)
        except Exception as e:
            print(len(name_list), reference_index)
            return []
    
    feature = []
    for name in name_list:
        feature.extend(transform_feature([string_utils.author_name_clean(name)], "name"))

    return feature

# generate the feature of title
def gen_feature_title(title):
    if title:
        return transform_feature(string_utils.clean_sentence(title, stemming=True), 'title')
    else:
        return []

# generate the feature of affiliation
def gen_feature_aff(aff):
    if aff:
        return transform_feature(string_utils.clean_sentence(aff), 'affiliation')
    else:
        return []

# generate the feature of venue
def gen_feature_venue(venue):
    if venue:
        return transform_feature(string_utils.clean_sentence(venue), 'venue')
    else:
        return []

# generate the feature of keywords
def gen_feature_keywords(keywords, title):
    if keywords == None or len(keywords) == 0:
        return transform_feature(string_utils.extractKeyword(title), 'keywords')
    else:
        return transform_feature([string_utils.author_name_clean(k) for k in keywords], 'keywords')

# generate the feature of abstract
def gen_feature_abstract(abstract, title):
    if abstract == None or len(abstract) == 0:
        return transform_feature(string_utils.clean_sentence(title, stemming=True), 'title')
    else:
        return transform_feature(string_utils.clean_sentence(abstract, stemming=True), 'abstract')

# gengerate the pair of papers，like [(paper_id0, paper_id1), (paper_id0, paper_id2), (paper_id0, paper_id3), ...]
def extract_complete_pub_pair_list(pub_list):
    pub_pair_list = []
    for index_1, pub_1 in enumerate(pub_list):
        for index_2, pub_2 in enumerate(pub_list):
            pub_pair_list.append((pub_1, pub_2))
    return pub_pair_list

# the class of the similarity of feature 
class FeatureSimilarityModel:
    def __init__(self):
        self.idf_dict = pickle.load(open(path_file.idf_dict_path, "rb"))
        self.model = load()

    # the main function
    def cal_pairwise_sim(self, raw_pub_list):
        pub_list = []
        for raw_pub in raw_pub_list:
            pub_list.append({
                'authors': gen_feature_author(raw_pub['authors'], raw_pub['reference_index']),
                'affiliation': gen_feature_aff(raw_pub['affiliation']),
                'title': gen_feature_title(raw_pub['title']),
                'venue': gen_feature_venue(raw_pub['venue']),
                'keywords': gen_feature_keywords(raw_pub['keywords'], raw_pub['title']),
                # 'abstract':  gen_feature_abstract(raw_pub['abstract'], raw_pub['title']),
                'year': raw_pub['year'],
            })
        pub_pair_list = extract_complete_pub_pair_list(pub_list)
        pairwise_sim_list = []
        print('Computing pairwise similarity... ({}/{})'.format(0, len(pub_pair_list)))
        for index, pub_pair in enumerate(pub_pair_list):
            if (index + 1) % 100000 == 0:
                print('Computing pairwise similarity... ({}/{})'.format(index + 1, len(pub_pair_list)))
            pairwise_sim_list.append(self.cal_sim(*pub_pair))
        print('Computing pairwise similarity... ({}/{})'.format(len(pub_pair_list), len(pub_pair_list)))
        print('Applying adaptive masking...')
        return pairwise_sim_list

    # the function to calculate the similarity of the pair of features
    def cal_sim(self, pub_1, pub_2):
        coauthor_sim = Similarity_function.coauthor_sim(pub_1['authors'], pub_2['authors'], self.idf_dict) 
        affiliation_sim = Similarity_function.affiliation_sim(pub_1['affiliation'], pub_2['affiliation'], self.idf_dict)
        venue_sim = Similarity_function.venue_sim(pub_1['venue'], pub_2['venue'], self.model, self.idf_dict)
        keywords_sim = Similarity_function.keywords_sim(pub_1['keywords'], pub_2['keywords'], self.idf_dict)
        title_sim = Similarity_function.title_abstract_sim(pub_1['title'] , pub_2['title'], self.model, self.idf_dict) # + pub_1['abstract']  + pub_2['abstract']
        # year_sim = Similarity_function.cal_year_sim(pub_1['year'], pub_2['year'])
        all_feature_sim = Similarity_function.all_feature_sim(pub_1['authors'] + pub_1['affiliation'] + pub_1['title'] + \
            pub_1['keywords'] + pub_1['venue'], pub_2['authors'] + pub_2['affiliation'] + pub_2['title'] + \
                pub_2['keywords'] + pub_2['venue'], self.idf_dict) # + Similarity_function.cal_year_sim(pub_1['year'], pub_2['year'])
        return [coauthor_sim, affiliation_sim, title_sim, venue_sim, keywords_sim, all_feature_sim]