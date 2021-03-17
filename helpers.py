from nltk.corpus import wordnet
import re


def compareParagraph(p1, p2):
    retval = 0
    biggest = len(p1) if len(p1) > len(p2) else len(p2)
    w = None
    while len(p1) > 0 and len(p2) > 0:
        val = -1
        for w2 in p2:
            temp = compareWords(p1[0], w2)
            if temp > val:
                val = temp
                w = w2
        retval = + val
        p1 = p1[1:]
        p2.remove(w)
    return retval / biggest


def compareWords(w1, w2):
    retval = -1

    for syn1 in wordnet.synsets(w1):
        for syn2 in wordnet.synsets(w2):
            temp = syn1.wup_similarity(syn2)
            if temp > retval:
                retval = temp

    return retval
