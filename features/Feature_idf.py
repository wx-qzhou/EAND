import json
import math
from utils import path_file, data_utils
from collections import defaultdict as dd
from features import Gen_features_similarity

'''
This code is used to generate the Inverse Document Frequency(IDF) of the words in the documents.
IDF = log(the total number of documents in the corpus / (1 + the number of the documents that contains the word))
'''

# fast
def read_raw_data(file_name):
    with open(file_name, 'r', encoding='UTF-8') as json_file:
        raw_data = json.load(json_file)
        return raw_data['pubs'], [raw_data['assignment'][group_id] for group_id in raw_data['assignment']]

# store the documents related to all author names
pub = {}
pubs = {}
for file_index, file_name in enumerate(path_file.train_raw_data_list + path_file.test_raw_data_list):
        pub, _ = read_raw_data(file_name)
        pubs = dict(list(pub.items()) + list(pubs.items()))
print(len(pubs))

# the words of the features are stored
features_list = []
authors_list = []
affiliation_list = []
title_list = []
venue_list = []
keywords_list = []
abstract_list = []
year_list = []

document_list = []
authors_list_all = []
affiliation_list_all = []
title_list_all = []
venue_list_all = []
keywords_list_all = []
abstract_list_all = []
year_list_all = []

for pub_id in pubs:
    # author name
    if 'authors' in pubs[pub_id]:
        authors = pubs[pub_id]['authors']
    else:
        authors = []
    authors_list = Gen_features_similarity.gen_feature_author(authors, pubs[pub_id]['reference_index'])
    authors_list_all += authors_list
    # author affiliation
    if 'affiliation' in pubs[pub_id]:
        affiliation = pubs[pub_id]['affiliation']
    else:
        affiliation = []
    affiliation_list = Gen_features_similarity.gen_feature_aff(affiliation)
    affiliation_list_all += affiliation_list
    # title
    if 'title' in pubs[pub_id]:
        title = pubs[pub_id]['title']
    else:
        title = []
    title_list = Gen_features_similarity.gen_feature_title(title)
    title_list_all += title_list
    # venue
    if 'venue' in pubs[pub_id]:
        venue = pubs[pub_id]['venue']
    else:
        venue = []
    venue_list = Gen_features_similarity.gen_feature_venue(venue)
    venue_list_all += venue_list
    # keywords
    if 'keywords' in pubs[pub_id]:
        keywords = pubs[pub_id]['keywords']
    else:
        keywords = []
    keywords_list = Gen_features_similarity.gen_feature_keywords(keywords)
    keywords_list_all += keywords_list
    # # abstract
    # if 'abstract' in pubs[pub_id]:
    #     abstract = pubs[pub_id]['abstract']
    # else:
    #     abstract = []
    # abstract_list = Gen_features_similarity.gen_feature_abstract(abstract)
    # abstract_list_all += abstract_list
    # # year
    # if 'year' in pubs[pub_id]:
    #     year = pubs[pub_id]['year']
    # else:
    #     year = []
    # year_list = [pubs[pub_id]['year']]
    # year_list_all += year_list

    features_list = authors_list + affiliation_list + title_list + venue_list + keywords_list + abstract_list + year_list
    document_list.append(features_list) # generate all documents list

features_set_all = set(authors_list_all + affiliation_list_all + title_list_all + venue_list_all + keywords_list_all + abstract_list_all + year_list_all)
print(len(features_set_all))

counter = dd(int) 
for d_list in document_list: # get the features of one document
    for f in set(d_list).intersection(features_set_all):
        counter[f] += 1  # counter = {feature: number}

print(len(counter))

cnt = len(document_list) # how many documents are there?
# 计算每个词的IDF
idf = dd(int)
for k in counter:
    idf[k] = math.log(cnt / (counter[k] + 1)) # idf = {feature： idf_value}

data_utils.dump_data(dict(idf), path_file.feature_path, "feature_idf.pkl")