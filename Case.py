import unicodedata
import re
import requests
import nltk
nltk.download('maxent_ne_chunker', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('words', quiet=True)
from bs4 import BeautifulSoup
from nltk import word_tokenize
from helpers import compareMultiSentences, removeFluff


class Case:

    def __init__(self, classifier, json):
        self.topic = []
        self.keywords = json["keywords"].split(" — ")
        self.date = json["judgmentDate"]
        self.url = "https://www.canlii.org" + json["path"]
        self.id = json["concatId"]
        self.title = json["title"]
        self._appellant = None
        self._respondent = None
        self.json = json
        self._classifier = classifier
        self._nono_words = ["TO", "LS", "DT", "CD", "IN", "NNP", "PRP", "POS", ":", "''", "CC", "``"]
        if "AT" in json and json["AT"]:
            self.advanceText = json["AT"]
            self._facts = json["AT"]["facts"]
            self._legislation = json["AT"]["legislation"]
            self._verdict = json["AT"]["verdict"]
            self._loaded = True
            self._citations = None
            self._sentences = None
        else:
            self.advanceText = None
            self._sentences = None
            self._facts = None
            self._legislation = None
            self._verdict = None
            self._analysis = None
            self._loaded = False
            self._citations = None

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
            if "We have established that your IP address has been used to access large amounts of documents posted on the CanLII website" in parsed_html.get_text():
                raise Exception("CanLII cut-off the Connection.")
            self.advanceText = parsed_html.get_text()
            return self.advanceText
        except requests.exceptions.ConnectionError as e:
            print(f"Invalid URL: {self.url}")
            raise e

    def _getSentences(self):
        if self._sentences:
            return self._sentences
        text = self._getAT()

        temp = text.split('Unfavourable mentions')

        if len(temp) == 1:
            temp = text
        else:
            temp = "".join(temp[1])

        temp = "".join(temp.split('ABOUT')[0])

        temp = temp.split("\n")

        info = []

        # This for loop separates citations and sentences with minimal edits
        for p in temp:
            p = unicodedata.normalize("NFKD", p)
            p = p.replace('\n', '')
            p = p.replace('\t', '')
            p = p.replace('\'', "'")
            p = p.replace("\\'", "'")
            if any(i in p for i in ['Copy text', 'Copy citation', 'Copy link', 'Citing documents']):
                continue
            p = p.strip()
            # Removes all whitespace indexes
            if "" == p:
                continue
            # Removes One-worded entrees
            if " " not in p:
                continue
            info.append(p)
        temp2 = " ".join(info)
        info = [""]
        temp2 = re.sub(r"[\(\[].*?[\)\]]", "", temp2)
        temp2 = re.sub(r".[\)]", "", temp2)
        tokens = word_tokenize(temp2)
        tags = nltk.pos_tag(tokens)
        for i in range(0, len(tags)):
            w = tags[i][0]
            tag = tags[i][1]
            # It's the end of a sentence when a period is present,
            isPeriod = tag == "."
            # not in front of a capital 'A.',
            isAbrev = tokens[i - 1][-1].isupper()
            # and it's followed by a space and another capital '. John'
            isMaybeEnd = i+1 == len(tokens) or tokens[i + 1][0].isupper()

            if isPeriod and not isAbrev and isMaybeEnd:
                info[-1] = info[-1].strip() + "."
                # print(info[-1])
                info.append("")
                continue

            if tag in [".", ":", ","]:
                info[-1] = info[-1].strip()

            info[-1] = info[-1] + w + " "

        self._sentences = info
        return info

    def getFacts(self):
        if self._facts:
            return self._facts

        sentences = self._getSentences()
        pastTense = []

        for s in sentences:
            words = word_tokenize(s)
            tags = nltk.pos_tag(words)
            #print(tags)
            for w in tags:
                if w[1] == "VBD":
                    # print(s)
                    #print(tags)
                    pastTense.append(s)
                    break

        facts = []
        for p in pastTense:
            #print(p)
            filtered_sentence = removeFluff(p)
            if not filtered_sentence:
                continue
            case_type, confidence = self._classifier.classify(filtered_sentence)
            # print(f"Type: {case_type}\tConfidence: {confidence}")
            if case_type.value == 2:
                facts.append(p)
        self._facts = facts
        return self._facts

    def getLegislation(self):
        if self._legislation:
            return self._legislation
        text = self._getAT()

        # Get the Legislations used
        self._legislation = []
        if "Legislation" in text:
            for i in text.split("Legislation")[1].split("Citation")[0].split("Decisions")[0].split("Help")[0].split(
                    "Acknowledgements")[0].split("\n\n"):
                i = unicodedata.normalize("NFKD", i)
                i = i.replace('\n', '')
                i = i.replace('\t', '')
                i = i.replace('\'', "'")
                i = i.strip()
                if i == "":
                    continue
                self._legislation.append(i.split(";")[0])

        return self._legislation

    def getDecision(self):
        if self._verdict:
            return self._verdict

        sentences = self._getSentences()
        check = []
        buzz_words = [
            "dismiss",
            "allow",
            "denied",
            "approve",
            "disposition",
            "guilty",
            "impose"
        ]

        # Check for buzz words
        for s in sentences:
            if any(w in s.lower() for w in buzz_words):
                check.append(s)

        if len(check) == 0:
            check = sentences

        retval = ""
        max = -1

        for p in check:
            filtered_sentence = removeFluff(p)
            if not filtered_sentence:
                continue
            votes = self._classifier.classify(filtered_sentence)
            isMajority = votes[1] > votes[0] and votes[1] > votes[2] and votes[1] > votes[3]
            isMax = votes[1] >= max
            if isMax and isMajority:
                print(f"New Max: {p}   Score: {votes[1]}")
                max = votes[1]
                retval = p
            else:
                print(f"Failed: {p}   Score: {votes[1]}")

        self._verdict = retval
        return self._verdict

    def getCitations(self):
        if self._citations:
            return self._citations
        text = self._getAT()

        # Get the citations used
        self._citations = []
        if "Decisions" in text:
            for i in text.split("Decisions")[1].split("Legislation")[0].split("Decisions")[0].split("Help")[0].split(
                    "Acknowledgements")[0].split("\n\n"):
                i = unicodedata.normalize("NFKD", i)
                i = i.replace('\n', '')
                i = i.replace('\t', '')
                i = i.replace('\'', "'")
                i = i.strip()
                if i == "":
                    continue
                self._citations.append(i.split(";")[0])

        return self._citations

    def getAppellant(self):
        if self._appellant:
            return self._appellant
        if "v." in self.title:
            self._appellant = self.title.split("v.")[0].strip()
        elif " v " in self.title:
            self._appellant = self.title.split(" v ")[0].strip()
        else:
            self._appellant = ""

        return self._appellant

    def getRespondent(self):
        if self._respondent:
            return self._respondent
        if "v." in self.title:
            self._respondent = self.title.split("v.")[1].strip()
        elif " v " in self.title:
            self._respondent = self.title.split(" v ")[1].strip()
        else:
            self._respondent = ""

        return self._respondent

    def compareFacts(self, Case2):
        return compareMultiSentences(self.getFacts(), Case2.getFacts())

    def compareLegislation(self, Case2):

        return compareMultiSentences(self.getLegislation(), Case2.getLegislation())

    def compareDecision(self, Case2):

        return compareMultiSentences(self.getDecision(), Case2.getDecision())