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
    class _HelperClassifier(ClassifierI):
        def labels(self):
            pass

        def __init__(self, name, c, reload):
            # This will get us the name of all subfolders in the DB #
            self.name = name
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
                    features_f = open("./PICKLE_FILES/" + name + "_features.pickle", "rb")
                    self.word_features = pickle.load(features_f)
                    features_f.close()
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
                            word = w[0].lower()
                            all_words.append(word)
                            if w[1] in nono_words:
                                continue
                            if len(word) == 1:
                                continue
                            if word not in stop_words:
                                filtered_sentence.append(word)
                        documents.append((filtered_sentence, d2 == d))
                        f.close()

                random.shuffle(documents)
                all_words = nltk.FreqDist(all_words)
                self.word_features = list(all_words.keys())[:2000]
                save_features = open("./PICKLE_FILES/" + name + "_features.pickle", "wb")
                pickle.dump(self.word_features, save_features)
                save_features.close()

                def find_features(doc):
                    words = set(doc)
                    features = {}
                    for w in self.word_features:
                        features[w] = (w in words)
                    return features
                feature_set = [(find_features(rev), category) for (rev, category) in documents]
                training_set = feature_set
                #self.training_set = feature_set[:int(len(feature_set))]  # Arbitrary nums
                #self.testing_set = feature_set[len(training_set):]  # Arbitrary nums

                #  This is stats
                classifiers[paths.index(d)] = classifiers[paths.index(d)].train(training_set)

                #print("Classifier: " + name + "_" + d)
                #print(nltk.classify.accuracy(classifiers[paths.index(d)], self.testing_set))
                # print(testing_set)



                save_classifier = open("./PICKLE_FILES/" + name + "_" + d + ".pickle", "wb")
                pickle.dump(classifiers[paths.index(d)], save_classifier)
                save_classifier.close()

            self.__classifiers = classifiers

        def calculate(self, wordList):
            retval = []
            for c in self.__classifiers:
                data = {}
                for w in wordList:
                    data.update({w: w in self.word_features})
                retval.append(c.classify(data))
            return retval

    def labels(self):
        pass

    def __init__(self, reload=False):
        classifiers = []
        # With 41 potential classifiers,
        # It's easier to list the classifiers that don't suit our needs over those that does
        # Some would fit, but require additional calls
        nono = [
            "CategoricalNB",
            "ClassifierChain",
            "GaussianNB",
            "GaussianProcessClassifier",
            "HistGradientBoostingClassifier",
            "LabelPropagation",
            "LabelSpreading",
            "LinearDiscriminantAnalysis",
            "LinearSVC",
            "LogisticRegressionCV",
            "MultiOutputClassifier",
            "NuSVC",
            "OneVsOneClassifier",
            "OneVsRestClassifier",
            "OutputCodeClassifier",
            "QuadraticDiscriminantAnalysis",
            "StackingClassifier",
            "VotingClassifier",
            "RadiusNeighborsClassifier"
        ]
        clas = [est for est in all_estimators() if issubclass(est[1], ClassifierMixin)]
        for c in clas:
            if c[0] in nono:
                continue
            #print(c[0])
            classifiers.append(self._HelperClassifier(c[0], SklearnClassifier(c[1]()), reload=reload))
        self.__classifiers = classifiers

    def classify(self, wordList):
        votes = [0, 0, 0, 0]
        for c in self.__classifiers:
            ret = c.calculate(wordList)
            assert len(ret) == 4
            if ret[0]:
                votes[0] += 1
            if ret[1]:
                votes[1] += 1
            if ret[2]:
                votes[2] += 1
            if ret[3]:
                votes[3] += 1
        for index, v in enumerate(votes):
            votes[index] = v/len(self.__classifiers)
        return votes
