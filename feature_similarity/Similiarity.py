from feature_similarity.Function import Cal_one_paper_tf_idf
from utils.args import THREAD_VALUE

"""get common elements"""
def common_elements(elements_list1, elements_list2):
    common_elements_list = list(set(elements_list1).intersection(set(elements_list2)))
    return common_elements_list

"""IDF of feature"""
def feature_IDF(feature_list1, feature_list2, IDF):
    common_feature_list = common_elements(feature_list1, feature_list2)
    
    if len(common_feature_list) == 0:
        return 0

    similarity = 0
    for all_feature in common_feature_list:
        similarity += IDF.get(all_feature, THREAD_VALUE) / (2 * THREAD_VALUE)
    return similarity

"""jac of feature"""
def feature_jac(feature_list1, feature_list2):
    if len(feature_list1) == 0 or len(feature_list2) == 0:
        return 0
    common_feature = len(common_elements(feature_list1, feature_list2))
    if min(len(feature_list1), len(feature_list2)) - common_feature == 0:
        return 1
    else:
        return common_feature / (len(feature_list1) + len(feature_list2) - common_feature)

"""dice of feature"""
def feature_dice(feature_list1, feature_list2):
    if len(feature_list1) == 0 or len(feature_list2) == 0:
        return 0
    common_feature = len(common_elements(feature_list1, feature_list2))
    return (2 * common_feature) / (len(feature_list1) + len(feature_list2))

"""the tf-IDF of feature"""
def feature_TF_IDF(feature_list1, feature_list2, IDF):
    if len(feature_list1) == 0 or len(feature_list2) == 0:
        return 0
    tf_idf1 = Cal_one_paper_tf_idf(feature_list1, THREAD_VALUE, IDF=IDF)
    tf_idf2 = Cal_one_paper_tf_idf(feature_list2, THREAD_VALUE, IDF=IDF)
    common_feature_list = common_elements(feature_list1, feature_list2)
    similarity = 0
    if len(common_feature_list) == 0:
        return 0
    for coauthor in common_feature_list:
        similarity += (tf_idf1.get(coauthor) + tf_idf2.get(coauthor))
    return similarity


# """IDF of feature"""
# def feature_IDF(feature_list1, feature_list2, IDF):
#     common_feature_list = common_elements(feature_list1, feature_list2)
    
#     if len(common_feature_list) == 0:
#         return 0

#     similarity = 0
#     for all_feature in common_feature_list:
#         similarity += IDF.get(all_feature, THREAD_VALUE) / (2 * THREAD_VALUE)
#     similarity = similarity * (len(common_feature_list) ** 0.5) / len(common_feature_list)
#     return similarity

# """jac of feature"""
# def feature_jac(feature_list1, feature_list2):
#     if len(feature_list1) == 0 or len(feature_list2) == 0:
#         return 0
#     common_feature = len(common_elements(feature_list1, feature_list2))
#     if min(len(feature_list1), len(feature_list2)) - common_feature == 0:
#         return 1
#     else:
#         return common_feature / (2 * min(len(feature_list1), len(feature_list2)) - common_feature)

# """dice of feature"""
# def feature_dice(feature_list1, feature_list2):
#     if len(feature_list1) == 0 or len(feature_list2) == 0:
#         return 0
#     common_feature = len(common_elements(feature_list1, feature_list2))
#     return (2 * common_feature) / (2 * min(len(feature_list1), len(feature_list2)))

# """the tf-IDF of feature"""
# def feature_TF_IDF(feature_list1, feature_list2, IDF):
#     if len(feature_list1) == 0 or len(feature_list2) == 0:
#         return 0
#     tf_idf1 = Cal_one_paper_tf_idf(feature_list1, THREAD_VALUE, IDF=IDF)
#     tf_idf2 = Cal_one_paper_tf_idf(feature_list2, THREAD_VALUE, IDF=IDF)
#     common_feature_list = common_elements(feature_list1, feature_list2)
#     similarity = 0
#     if len(common_feature_list) == 0:
#         return 0
#     for coauthor in common_feature_list:
#         similarity += (tf_idf1.get(coauthor) + tf_idf2.get(coauthor))
#     return similarity

# if __name__ == "__main__":
#     from utils import data_utils
#     from utils import path_file
#     IDF = data_utils.load_data(path_file.feature_path, "feature_idf.pkl")
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
#         "__VENUE__conference"
#     ]
#     print(len(string))
#     string1 = [
#         "__TITLE__new",
#         "__TITLE__method22",
#         "__TITLE__solv",
#         "__VENUE__vehicular",
#         "__VENUE__technology",
#         "__VENUE__conference"
#     ]
#     print(len(string1))
#     print(feature_jac(string, string1))
#     print(feature_jac(string, string))
#     print(feature_dice(string, string1))
#     print(feature_IDF(string, string1, IDF))
#     print(feature_TF_IDF(string, string, IDF))
#     print(feature_TF_IDF(string, string, IDF))