from Query import Query
from CaseTextClassifier import CaseTextClassifier


def main():
    # For now, we can put None as it won't get call
    c = CaseTextClassifier()
    # q = Query(c)
    print(c.classify(["had", "helped"]))
    # results = q.search()
    # for r in results:
        # r.getAnalysis()
        # r.getDecision()
        # r.getFacts()
        # r.getLegislation()
    # print(len(results))


if __name__ == "__main__":
    main()
