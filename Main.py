import time

from Query import Query
from CaseTextClassifier import CaseTextClassifier
import os
import glob


def testing():
    #sentence1 = ["I love cooking and cleaning"]
    #sentence2 = ["I dislike painting and movies"]
    #print(compareMultiSentences(sentence1, sentence2))
    #return
    #print(compare([sentence1], [sentence2]))
    reload = False
    # For now, we can put None as it won't get call
    # print(len(docs))
    # print(docs[0])
    # print(len(docs[0][0]))
    # print(len(movie_reviews.words()))
    if reload:
        files = glob.glob('./PICKLE_FILES/*')
        for f in files:
            os.remove(f)
    c = CaseTextClassifier()
    q = Query(c)
    # print(c.classify(["had", "helped"]))
    q.setTextSearch("assault")
    results = q.search()
    print("Starting 0v1")
    start_time = time.time()
    # r = results[4]
    # d = r._loadData()
    #for i in range(0, len(results)):
    #    print()
    #    print()
    #    print(f"{results[i].getAppellant()} vs {results[i].getRespondent()}")
    #    print(f"Retval: {results[i].getFacts()}")
    print(f"Retval: {results[0].compareFacts(results[1])}")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Starting 2v3")
    start_time = time.time()
    print(f"Retval: {results[2].compareFacts(results[3])}")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Starting 4v5")
    start_time = time.time()
    print(f"Retval: {results[4].compareFacts(results[5])}")
    print("--- %s seconds ---" % (time.time() - start_time))
    print("Starting 0v2")
    start_time = time.time()
    print(f"Retval: {results[0].compareFacts(results[2])}")
    print("--- %s seconds ---" % (time.time() - start_time))
    #   print(i)
    #   print()
    # print(3)
    # d = r.getDecision()
    # print(4)
    # f = r.getFacts()
    # l = r.getLegislation()
    # print("Decisions")
    # print(d)
    # print("Facts")
    # print(f)
    # print("Legislation")
    # print(l)


if __name__ == "__main__":
    testing()
