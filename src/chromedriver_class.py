import requests, time, re, json, pickle, os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from pymongo import MongoClient
import pandas as pd 


class ChromeDriver(object):
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.all_hike_urls = None

    def _load_urls(self, url_of_pageofurls):
        self.driver.get(url_of_pageofurls)
        time.sleep(4)
        while True:
            try:
                loadMoreButton = self.driver.find_element_by_xpath(
                    "//button[contains(text(), 'Show more trails')]"
                    )
                time.sleep(2)
                loadMoreButton.click()
                time.sleep(8)
            except Exception as e:
                print(e)
                html = self.driver.page_source
                return html

    
    def save_urls_to_txt(self, url, filename):
        ''' Fetch and save html from website.
        '''
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                html = f.read()
        else:
            html = self._load_urls(url)
            with open(filename, 'w') as f:
                f.write(html)
        return html


    def get_urls(self, url, filename):
        ''' Parse html for the trails' urls and save urls.
        '''
        html = self.save_urls_to_txt(url, filename)
        soup = self._make_soup(html)
        links = soup.find_all('a', itemprop='url')
        self.all_hike_urls = ['https://www.alltrails.com'+links[i]['href'] for i in range(len(links))]
        return self.all_hike_urls

    def save_urls_to_json(self, url, filename_html, filename_json):
        self.all_hike_urls = self.get_urls(url, filename_html)
        with open(filename_json, 'w', encoding='utf-8') as f:
            json.dump(self.all_hike_urls, f, ensure_ascii=False, indent=4)
            

    def load_reviews(self, url):
        while True:
            try:
                loadMoreButton = self.driver.find_element_by_xpath("//button[contains(text(), 'Show more reviews')]")
                loadMoreButton.click()
                time.sleep(5)
            except:
                html = self.driver.page_source
                return html

    def _make_soup(self, html):
        soup = BeautifulSoup(html, 'lxml')
        return soup

