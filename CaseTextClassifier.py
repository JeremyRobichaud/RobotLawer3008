import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
import os
from os import path
import glob
import random
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier

# These are all grading algs that I could look into
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# This is for a custom classifer
from nltk.classify import ClassifierI
from statistics import mode


class CaseTextClassifier:
    class _HelperClassifier(ClassifierI):
        def labels(self):
            pass

        def __init__(self, name, c, reload):
            # This will get us the name of all subfolders in the DB #
            classifiers = []
            paths = []
            for x in os.walk("./DB"):
                if "\\" in x[0]:
                    paths.append(x[0].split("\\")[1])
                    classifiers.append(c)

            assert len(paths) == len(classifiers)

            # For each text type
            for d in paths:
                # If previously ran, use that
                if path.exists("./PICKLE_FILES/" + name + "_" + d + ".pickle") and not reload:
                    classifier_f = open("./PICKLE_FILES/" + name + "_" + d + ".pickle", "rb")
                    classifier = pickle.load(classifier_f)
                    classifier_f.close()
                    classifiers[paths.index(d)] = classifier
                    continue

                documents = []
                all_words = []
                for d2 in paths:
                    for p in glob.glob("./DB/" + d2 + "/*.txt"):
                        f = open(p, "r", encoding="utf8")
                        data = f.read()
                        stop_words = set(stopwords.words("english"))
                        words = word_tokenize(data)
                        pos = nltk.pos_tag(words)
                        nono_words = ["TO", "LS", "DT", "CD", "IN", "NNP", "PRP", "POS", ":", "''", "CC", "``"]
                        filtered_sentence = []
                        for w in pos:
                            if w[1] in nono_words:
                                continue
                            word = w[0].lower()
                            if len(word) == 1:
                                continue

                            all_words.append(word)
                            if word not in stop_words:
                                filtered_sentence.append(word)
                        documents.append((filtered_sentence, d2 == d))
                        f.close()

                random.shuffle(documents)
                all_words = nltk.FreqDist(all_words)
                word_features = list(all_words.keys())

                def find_features(doc):
                    words = set(doc)
                    features = {}
                    for w in word_features:
                        features[w] = (w in words)
                    return features

                feature_set = [(find_features(rev), category) for (rev, category) in documents]

                training_set = feature_set
                # training_set = feature_set[:int(len(feature_set)/2)]  # Arbitrary nums
                # testing_set = feature_set[len(training_set):]  # Arbitrary nums

                #  This is stats
                classifiers[paths.index(d)] = classifiers[paths.index(d)].train(training_set)

                # print("Classifier: " + name + "_" + d)
                # print(nltk.classify.accuracy(classifiers[paths.index(d)], testing_set))
                # print(testing_set)



                save_classifier = open("./PICKLE_FILES/" + name + "_" + d + ".pickle", "wb")
                pickle.dump(classifiers[paths.index(d)], save_classifier)
                save_classifier.close()

            self.__classifiers = classifiers

        def calculate(self, wordList):
            retval = []
            for c in self.__classifiers:
                dataT = {}
                dataF = {}
                accT = []
                accF = []
                for w in wordList:
                    dataT[w]: True
                    dataF[w]: False
                    accT.append(({w: True}, True))
                    accF.append(({w: False}, False))

                retT = (c.classify(dataT), (nltk.classify.accuracy(c, accT)))
                retF = (c.classify(dataF), (nltk.classify.accuracy(c, accF)))
                retval.append(retT if retT[1] > retF[1] else retF)
            return retval

    def labels(self):
        pass

    def __init__(self, reload=False):
        classifiers = [
            self._HelperClassifier("Bayes", nltk.NaiveBayesClassifier, reload=reload),
            self._HelperClassifier("MultiNomial", SklearnClassifier(MultinomialNB()), reload=reload),
            self._HelperClassifier("Bernoulli", SklearnClassifier(BernoulliNB()), reload=reload)
        ]
        self.__classifiers = classifiers

    def classify(self, wordList):
        votes = [0, 0, 0, 0]
        for c in self.__classifiers:
            ret = c.calculate(wordList)
            # print(ret)
            for index, b in enumerate(ret):
                if b[0]:
                    votes[index] = + b[1]
        for index, v in enumerate(votes):
            votes[index] = v/len(self.__classifiers)
        return votes
