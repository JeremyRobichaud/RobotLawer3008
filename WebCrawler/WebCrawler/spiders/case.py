import scrapy

class CaseSpider(scrapy.Spider):
    name = 'case'
    start_urls = [
            'https://www.canlii.org/en/search/type=decision&jId=ca,nb',
        ]

    def parse(self, response):
        filename = "test.html"
        with open(filename, "wb") as f:
            searchResults = response.xpath('//*[@id="searchResults"]')
            print(searchResults)
            print(response)
            f.write(response.body)
