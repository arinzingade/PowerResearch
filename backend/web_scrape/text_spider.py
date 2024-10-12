import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy import signals
from scrapy.signalmanager import dispatcher
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextSpider(scrapy.Spider):
    name = 'text_spider'

    def __init__(self, start_url, *args, **kwargs):
        super(TextSpider, self).__init__(*args, **kwargs)
        self.start_urls = [start_url]
        self.scraped_data = []

    def parse(self, response):
        logging.info(f"Parsing URL: {response.url}")
        # Extract all text from the page
        page_text = response.xpath('//body//text()').getall()
        page_text = [text.strip() for text in page_text if text.strip()]

        if not page_text:
            logging.warning(f"No text found on the page: {response.url}")
        
        # Add the extracted text and URL to the scraped data
        self.scraped_data.append({
            'url': response.url,
            'text': page_text
        })
        logging.info(f"Scraped data length: {len(page_text)} elements")

def spider_results(signal, sender, item, response, spider):
    logging.info(f"Spider {spider.name} scraped an item")

def spider_error(failure, response, spider):
    logging.error(f"Spider {spider.name} encountered an error on {response.url}: {failure}")

def run_spider(start_url):
    results = []

    def spider_closed(spider):
        logging.info(f"Spider {spider.name} closed. Scraped data: {spider.scraped_data}")
        results.extend(spider.scraped_data)

    dispatcher.connect(spider_results, signal=signals.item_scraped)
    dispatcher.connect(spider_error, signal=signals.spider_error)
    dispatcher.connect(spider_closed, signal=signals.spider_closed)

    process = CrawlerProcess(settings={
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO'
    })
    
    process.crawl(TextSpider, start_url=start_url)
    process.start()

    return results

# Example usage
if __name__ == '__main__':
    url = 'https://www.investopedia.com/apple-stock-gets-jefferies-downgrade-as-iphone-expectations-deemed-too-high-8724140'
    scraped_json = run_spider(url)
    logging.info(f"Scraped data: {scraped_json}")
    print(scraped_json)