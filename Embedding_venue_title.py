from os.path import join
import math
from collections import defaultdict as dd
from datetime import datetime
from utils.data_utils import read_raw_data, save, load_json
from utils import path_file
from feature_similarity.Embedding import word_2_vec_embedding
from features.Gen_features_similarity import gen_feature_title, gen_feature_venue, gen_feature_author, gen_feature_aff, gen_feature_keywords

start_time = datetime.now()

"""extract all feature which is inneed."""
def extract_feature(paper):
    paper_feature_list = []

    paper_feature_list += gen_feature_author(paper["authors"], paper["reference_index"])
    paper_feature_list += gen_feature_aff(paper["affiliation"])
    paper_feature_list += gen_feature_title(paper["title"])
    # paper_feature_list += gen_feature_venue(paper["venue"])
    paper_feature_list += gen_feature_keywords(paper["keywords"], paper['title'])

    return paper_feature_list


"""generate the feature of title and venue which are used by word2vec"""
def dump_author_features_to_file():
    pubs_dict = {}
    for i, raw_data in enumerate(path_file.train_raw_data_list + path_file.test_raw_data_list):
        pub = read_raw_data(raw_data)[0]
        if i % 1000 == 0:
            print(i, datetime.now()-start_time)
        pub_temp = {}
        for pid in pub:
            paper_feature_list = extract_feature(pub[pid])
            pubs_dict.update({pid : paper_feature_list})

    print('n_papers', len(pubs_dict))
    save(pubs_dict, path_file.preprocessed_path, "features")

def embedding():
    sentence = list(load_json(path_file.preprocessed_path, "features.json").values())
    word_2_vec_embedding(sentence)

if __name__ == '__main__':
    dump_author_features_to_file()
    embedding()