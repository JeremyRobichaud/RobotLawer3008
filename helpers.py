import nltk
from nltk.corpus import wordnet
import numpy
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.optimize import linear_sum_assignment
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('words', quiet=True)
nltk.download('wordnet', quiet=True)


def penn_to_wn(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    elif tag.startswith('V'):
        return wordnet.VERB
    return None


def removeFluff(sentence, wn=False):
    words = word_tokenize(sentence)
    filtered_sentence = []
    pos = pos_tag(words)
    for w in pos:
        if w[1] in ["TO", "LS", "DT", "CD", "IN", "NNP", ".", "POS", ":", "''", "CC", "``"]:
           continue
        word = w[0].lower()
        if len(word) == 1:
            continue
        if word not in set(stopwords.words("english")):
            if wn:
                wn_tag = penn_to_wn(w[1])
                if not wn_tag:
                    continue
                lemmatzr = WordNetLemmatizer()
                lemma = lemmatzr.lemmatize(w[0], pos=wn_tag)
                syns = wordnet.synsets(lemma, pos=wn_tag)
                if syns:
                    filtered_sentence.append(syns[0])
            else:
                filtered_sentence.append(word)
    return filtered_sentence


def _compareMatrix(rows, columns, func):
    matrix = numpy.array([[(func(c, r) * -1) for c in columns] for r in rows])
    # This uses the Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(matrix)
    return (matrix[row_ind, col_ind].sum() * -1) / (len(rows) if len(rows) < len(columns) else len(columns))


def compareSentences(p1, p2):
    corpus = [p1, p2]
    vectorizer = CountVectorizer(ngram_range=(1, 3))
    vectorizer.fit(corpus)

    x1 = vectorizer.transform([p1])
    x2 = vectorizer.transform([p2])
    return cosine_similarity(x1, x2)[0][0]


def compareMultiSentences(t1, t2):
    return _compareMatrix(t1, t2, lambda x, y: compareSentences(x, y))
