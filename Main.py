from Query import Query
from CaseTextClassifier import CaseTextClassifier

def main():
    # For now, we can put None as it won't get call
    # print(len(docs))
    # print(docs[0])
    # print(len(docs[0][0]))
    # print(len(movie_reviews.words()))
    c = CaseTextClassifier()
    q = Query(c)
    # print(c.classify(["had", "helped"]))
    q.setTextSearch("murder")
    results = q.search()
    r = results[3]
    print(r.getAnalysis())
    print(r.getDecision())
    print(r.getFacts())
    print(r.getLegislation())
    print(len(results))


if __name__ == "__main__":

    main()
