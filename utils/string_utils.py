import nltk
import string
from nltk.corpus import stopwords 

def is_download_packages(is_download = False):
    if is_download:
        nltk.set_proxy('SYSTEM PROXY')
        from nltk.tokenize import word_tokenize
        nltk.download('stopwords')
        nltk.download('punkt')

stop = set(stopwords.words('english'))

punct = set(u''':!),.:;?.]}¢'"、。〉》」』〕〗〞︰︱︳﹐､﹒
﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂﹄﹏､￠
々‖•·ˇˉ―′’”([{£¥'"‵〈《「『〔〖（［｛￡￥〝︵︷︹︻
︽︿﹁﹃﹙﹛﹝（｛“‘_…/''')

# stopwords
stop_words = set(["a", "about", "again", "all", "almost", "also", "although", "always", "among", "an", "and", "another", "any", "are", "as", "at"
            , "be", "because", "been", "before", "being", "between", "both", "but", "by", "can", "could", "on", "did", "do", "does", "done"
            , "due", "during", "each", "either", "enough", "epecially", "etc", "for", "found", "from", "further", "had", "has", "have"
            , "having", "here", "how", "however", "i", "if", "in", "into", "is", "its", "itself", "just",  "becase",  "become", "kg", "km"
            , "made", "mainly", "make", "may", "mg", "might", "ml", "mm", "most", "mostly", "must", "nearly", "neither", "no", "nor"
            , "obtained", "of", "often", "on", "our", "overall", "perhaps", "pmid", "quite", "rather", "really", "regarding"
            , "seem", "seen", "several", "should", "show", "showed", "shown", "shows", "significantly", "since", "so", "some", "such"
            , "than", "that", "the", "their", "theirs", "them", "then", "there", "therefore", "these", "they", "this", "those", "through"
            , "thus", "to", "upon", "use", "used", "using", " ", "various", "very", ", ", ".", "d", "e", "f", "g",  "h", "j", "k"
            , "l", "m", "n", "o", "p", "r", "s", "t", "u", "v", "w", "x", "y", "z", "was", "we", "were", "what", "when", "which"
            , "while", "with", "within", "without", "would", "-", "b", "c"])

stemmer = nltk.stem.PorterStemmer()

# clean the sentence
def clean_sentence(text, stemming=False):
    text = text.lower()
    # clear punctuations
    # for token in string.punctuation:
    #     text = text.replace(token, "")
    for token in punct:
        text = text.replace(token, "")
    # split the word string into words list
    words = nltk.word_tokenize(text) 
    # clear the stopwords in the words list
    filter_words = [w for w in words if w not in stopwords.words('english')]
    filter_words = [w for w in filter_words if w not in stop_words]
    # generate the stem of words
    if stemming:
        stemmed_words = []
        for w in filter_words:
            stemmed_words.append(stemmer.stem(w))
        filter_words = stemmed_words
    # let the list be string
    # filter_words = ' '.join(filter_words)
    return filter_words

def clean_name(name):
    if name is None:
        return ""
    x = [k.strip() for k in name.lower().strip().replace(".", " ").replace("-", " ").split()]
    return "_".join(x)

# clear the author name
def author_name_clean(name):
    if name is None or len(name) == 0:
        return ""
    name = name[:-1].replace('. -','_').replace('  ','_').replace(' -', '_').replace('. ', '_').replace('.-', '_').replace('- ', '_')\
        .replace('.', '_').replace('-', '_').replace(' ', '_').lower() + name[-1].replace('-', '').replace('_', '').replace('.', '').lower()
    return name

import jieba.analyse

# get keywords
def extractKeyword(document, keyword_num=4):
    document = ' '.join(clean_sentence(document))
    return jieba.analyse.extract_tags(document, topK=5)

# if __name__ == "__main__":
#     s = "I love China, becase she is my country. And she has become more and more strong.『"
#     print(clean_sentence(s))
#     print(extractKeyword(s))