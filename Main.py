from Query import Query
from CaseTextClassifier import CaseTextClassifier
import os
import glob

def main():
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
    # r = results[4]
    # d = r._loadData()
    for i in range(0, len(results)):
        print()
        print()
        print(f"{results[i].getAppellant()} vs {results[i].getRespondent()}")
        print(f"Retval: {results[i].getFacts()}")
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
    main()
