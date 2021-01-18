import json
import urllib.request
from Case import Case


class Query:

    def __init__(self, classifier):
        self.type = "decision"
        self.jId = "ca,nb"
        self.text = None
        self.id = None
        self.maxPage = 4
        self._classifier = classifier

    def setTextSearch(self, text):
        self.text = text

    def setID(self, param):
        self.id = param

    def setJiD(self, param):
        self.jId = param

    def setType(self, param):
        self.type = param

    def setMaxPage(self, maxPage):
        if isinstance(maxPage, int) or maxPage > 4:
            return
        self.maxPage = maxPage

    def search(self):
        return self._searchPage(1)

    def _searchPage(self, pageNum):
        if pageNum > self.maxPage:
            return []
        url = "https://www.canlii.org/en/search/ajaxSearch.do?"
        url += f"type={self.type}&jId={self.jId}"
        url += f"&page={pageNum}"
        if self.text:
            url += f"&text={self.text}"
        if self.id:
            url += f"&id={self.id}"
        data = json.loads(urllib.request.urlopen(url).read().decode())
        results = data["results"]

        retval = []
        for res in results:
            retval.append(Case(self._classifier, res))

        if len(retval) < 25:
            return retval

        return retval + self._searchPage(pageNum + 1)
