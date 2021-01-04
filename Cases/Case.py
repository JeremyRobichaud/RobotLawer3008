class Case:

    def __init__(self, json):
        self.keywords = json["keywords"].split(" — ")
        self.date = json["judgmentDate"]
        self.url = "https://www.canlii.org" + json["path"]
        self.id = json["concatId"]
        self.title = json["title"]
        self.json = json

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
