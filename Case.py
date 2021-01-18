import requests
import nltk
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from helpers import compareParagraph


class Case:

    def __init__(self, classifier, json):
        self.keywords = json["keywords"].split(" — ")
        self.date = json["judgmentDate"]
        self.url = "https://www.canlii.org" + json["path"]
        self.id = json["concatId"]
        self.title = json["title"]
        self.json = json
        self._classifier = classifier
        self._nono_words = ["TO", "LS", "DT", "CD", "IN", "NNP", "PRP", "POS", ":", "''", "CC", "``"]
        if "AT" in json and json["AT"]:
            self.advanceText = json["AT"]
            self._facts = json["AT"]["facts"]
            self._legislation = json["AT"]["legislation"]
            self._verdict = json["AT"]["verdict"]
        else:
            self.advanceText = None
            self._facts = None
            self._legislation = None
            self._verdict = None
            self._analysis = None

    def getKeywords(self):
        return self.keywords

    def getDate(self):
        return self.date

    def getURL(self):
        return self.url

    def getID(self):
        return self.id

    def getTitle(self):
        return self.title

    def getJSON(self):
        return self.json

    def _getAT(self):
        if self.advanceText:
            return self.advanceText
        try:
            page = requests.get(self.url)
            parsed_html = BeautifulSoup(page.content, "lxml")
            self.advanceText = parsed_html.get_text()
            return self.advanceText
        except requests.exceptions.ConnectionError as e:
            print(f"Invalid URL: {self.url}")
            raise e

    def getFacts(self):
        if self._facts:
            return self._facts
        text = self._getAT()
        paragraphs = text.split('\n\n')
        facts = []
        for p in paragraphs:

            words = word_tokenize(p)
            filtered_sentence = []
            pos = nltk.pos_tag(words)
            for w in pos:
                if w[1] in self._nono_words:
                    continue
                word = w[0].lower()
                if len(word) == 1:
                    continue
                if word not in set(stopwords.words("english")):
                    filtered_sentence.append(word)

            if not filtered_sentence:
                continue
            votes = self._classifier.classify(filtered_sentence)
            if votes[2] > 0:
                print(votes)
                facts.append(p)
        self._facts = facts
        return self._facts

    def getLegislation(self):
        if self._legislation:
            return self._legislation
        text = self._getAT()
        paragraphs = text.split('\n\n')
        legislation = []
        for p in paragraphs:
            words = word_tokenize(p)
            filtered_sentence = []
            pos = nltk.pos_tag(words)
            for w in pos:
                if w[1] in self._nono_words:
                    continue
                word = w[0].lower()
                if len(word) == 1:
                    continue
                if word not in set(stopwords.words("english")):
                    filtered_sentence.append(word)

            if not filtered_sentence:
                continue
            votes = self._classifier.classify(filtered_sentence)
            if votes[3] > 0:
                print(votes)
                legislation.append(p)
        self._legislation = legislation
        return self._legislation

    def getDecision(self):
        if self._verdict:
            return self._verdict
        text = self._getAT()
        paragraphs = text.split('\n\n')
        verdict = []
        for p in paragraphs:

            words = word_tokenize(p)
            filtered_sentence = []
            pos = nltk.pos_tag(words)
            for w in pos:
                if w[1] in self._nono_words:
                    continue
                word = w[0].lower()
                if len(word) == 1:
                    continue
                if word in set(stopwords.words("english")):
                    filtered_sentence.append(word)

            if not filtered_sentence:
                continue
            votes = self._classifier.classify(filtered_sentence)
            if votes[1] > 0:
                # print(votes)
                verdict.append(p)
        self._verdict = verdict
        return self._verdict

    def getAnalysis(self):
        if self._analysis:
            return self._analysis
        text = self._getAT()
        paragraphs = text.split('\n\n')
        analysis = []
        for p in paragraphs:
            words = word_tokenize(p)
            filtered_sentence = []
            pos = nltk.pos_tag(words)
            for w in pos:
                if w[1] in self._nono_words:
                    continue
                word = w[0].lower()
                if len(word) == 1:
                    continue
                if word not in set(stopwords.words("english")):
                    filtered_sentence.append(word)

            if not filtered_sentence:
                continue
            votes = self._classifier.classify(filtered_sentence)
            if votes[0] > 0:
                print(votes)
                analysis.append(p)
        self._analysis = analysis
        return self._analysis

    def compareCases(self, Case2):
        def selfCompare(t1, t2):
            biggest = len(t1) if len(t1) > len(t2) else len(t2)
            retval = 0
            while len(t1) > 0 and len(t2) > 0:
                a1 = t1[0]
                a2 = None
                val = -1
                for a in t2:
                    temp = compareParagraph(a1, a)
                    if temp > val:
                        val = temp
                        a2 = a
                retval = + val
                t1.remove(a1)
                t2.remove(a2)
            return retval/biggest

        comparison = [0.0, 0.0, 0.0, 0.0]

        # Compare Analysis #
        comparison[0] = selfCompare(self.getAnalysis(), Case2.getAnalysis())

        # Compare Decision #
        comparison[1] = selfCompare(self.getDecision(), Case2.getDecision())

        # Compare Facts #
        comparison[2] = selfCompare(self.getFacts(), Case2.getFacts())

        # Compare Legislation #
        comparison[3] = selfCompare(self.getLegislation(), Case2.getLegislation())

        return comparison
