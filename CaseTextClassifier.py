import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
import os
from os import path
from enum import Enum
import glob
import random
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.base import ClassifierMixin
from sklearn.utils import all_estimators

# These are all grading algs that I could look into
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# This is for a custom classifer
from nltk.classify import ClassifierI
from statistics import mode


class CaseTextClassifier:
    class TextClassification(Enum):
        ANALYSIS = 0
        DECISION = 1
        FACTS = 2
        LEGISLATION = 3
        OTHER = 4

    def __init__(self):
        self._labels = ["analysis", "decision", "facts", "legislation"]
        self._paths = self._getPaths()
        self._word_features = self._getWordFeatures()
        self._nb_training, self._nb_testing = self._getNonBinarySets()
        self._b_training, self._b_testing = self._getBinarySets()
        self._nb_classifiers = self._trainNonBinary()
        self._b_classifiers = self._trainBinary()

    def _find_features(self, doc):
        words = set(doc)
        features = {}
        for w in self._word_features:
            features[w] = (w in words)
        return features

    @staticmethod
    def _getPaths():
        paths = []
        for x in os.walk("./DB"):
            if "\\" in x[0]:
                paths.append(x[0].split("\\")[1])
        return paths

    def _getWordFeatures(self):
        if path.exists("./PICKLE_FILES/word_features.pickle"):
            classifier_f = open("./PICKLE_FILES/word_features.pickle", "rb")
            features = pickle.load(classifier_f)
            classifier_f.close()
            return features
        # This will get us the name of all subfolders in the DB #

        all_words = []
        for d2 in self._paths:
            for p in glob.glob("./DB/" + d2 + "/*.txt"):
                f = open(p, "r", encoding="utf8")
                data = f.read()
                words = word_tokenize(data)
                pos = nltk.pos_tag(words)
                for w in pos:
                    word = w[0].lower()
                    all_words.append(word)
                f.close()

        all_words = nltk.FreqDist(all_words)
        word_features = list(all_words.keys())[:2000]

        save_features = open("./PICKLE_FILES/word_features.pickle", "wb")
        pickle.dump(word_features, save_features)
        save_features.close()
        return word_features

    def _getNonBinarySets(self):
        if path.exists("./PICKLE_FILES/nb_feature_set.pickle"):
            feature_set_f = open("./PICKLE_FILES/nb_feature_set.pickle", "rb")
            feature_set = pickle.load(feature_set_f)
            feature_set_f.close()
            training_set = feature_set[:int(len(feature_set) * 3 / 4)]  # Arbitrary nums
            testing_set = feature_set[int(len(feature_set) * 3 / 4):]  # Arbitrary nums

            return training_set, testing_set


        documents = []
        for d2 in self._paths:
            for p in glob.glob("./DB/" + d2 + "/*.txt"):
                f = open(p, "r", encoding="utf8")
                data = f.read()
                stop_words = set(stopwords.words("english"))
                words = word_tokenize(data)
                pos = nltk.pos_tag(words)
                nono_words = ["TO", "LS", "DT", "CD", "IN", "NNP", "PRP", "POS", ":", "''", "CC", "``"]
                filtered_sentence = []
                for w in pos:
                    word = w[0].lower()
                    if w[1] in nono_words:
                        continue
                    if len(word) == 1:
                        continue
                    if word not in stop_words:
                        filtered_sentence.append(word)
                documents.append((filtered_sentence, d2))
                f.close()

            random.shuffle(documents)

        feature_set = [(self._find_features(rev), category) for (rev, category) in documents]
        save_feature_set = open("./PICKLE_FILES/nb_feature_set.pickle", "wb")
        pickle.dump(feature_set, save_feature_set)
        save_feature_set.close()

        training_set = feature_set[:int(len(feature_set)*3/4)]  # Arbitrary nums
        testing_set = feature_set[int(len(feature_set)*3/4):]  # Arbitrary nums

        return training_set, testing_set

    def _getBinarySets(self):
        ret_train = []
        ret_test = []
        if path.exists("./PICKLE_FILES/b_feature_set.pickle"):
            feature_set_f = open("./PICKLE_FILES/b_feature_set.pickle", "rb")
            feature_set = pickle.load(feature_set_f)
            feature_set_f.close()
        else:
            # TODO: Split into 4 list
            documents = []
            for d in self._paths:
                for d2 in self._paths:
                    for p in glob.glob("./DB/" + d2 + "/*.txt"):
                        f = open(p, "r", encoding="utf8")
                        data = f.read()
                        stop_words = set(stopwords.words("english"))
                        words = word_tokenize(data)
                        pos = nltk.pos_tag(words)
                        nono_words = ["TO", "LS", "DT", "CD", "IN", "NNP", "PRP", "POS", ":", "''", "CC", "``"]
                        filtered_sentence = []
                        for w in pos:
                            word = w[0].lower()
                            if w[1] in nono_words:
                                continue
                            if len(word) == 1:
                                continue
                            if word not in stop_words:
                                filtered_sentence.append(word)
                        documents.append((filtered_sentence, d==d2))
                        f.close()

            random.shuffle(documents)

            feature_set = [(self._find_features(rev), category) for (rev, category) in documents]
            save_feature_set = open("./PICKLE_FILES/b_feature_set.pickle", "wb")
            pickle.dump(feature_set, save_feature_set)
            save_feature_set.close()

        training_set = feature_set[:int(len(feature_set)*3/4)]  # Arbitrary nums
        testing_set = feature_set[int(len(feature_set)*3/4):]  # Arbitrary nums

        ret_train.append(training_set)
        ret_test.append(testing_set)

        return ret_train, ret_test

    def _trainNonBinary(self):
        retval = []
        names = ["MultiNB"]
        classifiers = [SklearnClassifier(MultinomialNB())]

        for i in range(0, len(names)):
            n = names[i]
            c = classifiers[i]
            if path.exists("./PICKLE_FILES/" + n + ".pickle"):
                multiNB_f = open("./PICKLE_FILES/" + n + ".pickle", "rb")
                multiNB = pickle.load(multiNB_f)
                multiNB_f.close()
                retval.append(multiNB)
            else:
                classifier = c.train(self._nb_training)
                save_classifier = open("./PICKLE_FILES/" + n + ".pickle", "wb")
                pickle.dump(classifier, save_classifier)
                save_classifier.close()
                retval.append(classifier)

        return retval

    def _trainBinary(self):
        retval = []
        names = ["BernoulliNB"]
        classifiers = [SklearnClassifier(BernoulliNB())]

        for i in range(0, len(names)):
            n = names[i]
            c = classifiers[i]
            council = []
            for p in self._paths:
                if path.exists("./PICKLE_FILES/" + n + "_" + p + ".pickle"):
                    classifier_f = open("./PICKLE_FILES/" + n + "_" + p + ".pickle", "rb")
                    classifier = pickle.load(classifier_f)
                    classifier_f.close()
                    council.append(classifier)
                else:
                    # TODO: Fix this
                    classifier = c.train(self._b_training[i])
                    save_classifier = open("./PICKLE_FILES/" + n + "_" + p + ".pickle", "wb")
                    pickle.dump(classifier, save_classifier)
                    save_classifier.close()
                    council.append(classifier)
            retval.append(council)

        return retval

    def classify(self, wordList):
        votes = [0, 0, 0, 0]
        data = {}
        for w in wordList:
            data.update({w: w in self._word_features})
        for cl in self._b_classifiers:
            ret = [c.classify(data) for c in cl]
            print(len(cl))
            print(len(self._b_testing))
            accuracy = [nltk.classify.accuracy(cl[i], self._b_testing[i]) for i in range(0, len(cl))]
            assert len(ret) == 4
            print(ret)
            print(accuracy)
            votes[0] += accuracy[0] if ret[0] else 1-accuracy[0]
            votes[1] += accuracy[1] if ret[1] else 1-accuracy[1]
            votes[2] += accuracy[2] if ret[2] else 1-accuracy[2]
            votes[3] += accuracy[3] if ret[3] else 1-accuracy[3]

        for cl in self._nb_classifiers:
            ret = cl.classify(data)
            assert ret in self._labels
            accuracy = nltk.classify.accuracy(cl, self._nb_testing)
            votes[self._labels.index(ret)] += accuracy

        # TODO: Adjust Math
        for index, v in enumerate(votes):
            votes[index] = v/len(self._nb_classifiers + self._b_classifiers)
        print(f"Votes: {votes}")
        max = -1
        maxI = -1
        for index, v in enumerate(votes):
            if v <= max:
                continue
            max = v
            maxI = index

        return CaseTextClassifier.TextClassification(maxI), max
