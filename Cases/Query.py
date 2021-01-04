import json
import urllib.request
import requests
from lxml import html
from bs4 import BeautifulSoup
from Cases.Case import Case


class Query:

    def __init__(self):
        self.type = "decision"
        self.jId = "ca,nb"
        self.text = None
        self.id = None
        self.maxPage = 8727

    def setTextSearch(self, text):
        self.text = text

    def setID(self, param):
        self.id = param

    def setMaxPage(self, maxPage):
        self.maxPage = maxPage

    def search(self):
        return self._searchPage(1)

    def _searchPage(self, pageNum):
        if pageNum > self.maxPage:
            return []
        url = "https://www.canlii.org/en/search/ajaxSearch.do?"
        url += "type=decision&jId=ca,nb"
        url += f"&page={pageNum}"
        if self.text:
            url += f"&text={self.text}"
        if self.id:
            url += f"&id={self.id}"
        data = json.loads(urllib.request.urlopen(url).read().decode())
        results = data["results"]

        retval = []
        for res in results:
            retval.append(Case(res))

        if len(retval) < 25:
            return retval

        return retval + self._searchPage(pageNum + 1)

def main():
    q = Query()
    q.setMaxPage(2)
    results = q.search()
    for res in results:
        print(res.getURL())
    return
    webUrl = urllib.request.urlopen('https://www.canlii.org/en/ca/scc/doc/2008/2008scc9/2008scc9.html')

    print("result code: " + str(webUrl.getcode()))

    # read the data from the URL and print it
    data = webUrl.read()
    myString = data.decode("utf8")
    webUrl.close()
    #print(myString)

    page = requests.get('https://www.canlii.org/en/ca/scc/doc/2008/2008scc9/2008scc9.html')
    parsed_html = BeautifulSoup(page.content, "lxml")
    tree = html.fromstring(page.content)
    section20 = parsed_html.body.find('div', attrs={'class': 'Section20'})
    print(section20)



if __name__ == "__main__":
    main()
