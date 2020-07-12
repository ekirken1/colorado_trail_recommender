import requests, time, re, json, pickle, os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
import pandas as pd 


class ChromeDriver(object):
    def __init__(self):
        self.driver = webdriver.Chrome()
        self.all_hike_urls = None

    
    def _login_alltrails(self, login_url, username, password):
        self.driver.get(login_url)
        # maybe try logging into google, then alltrails continue with google

        password = self.driver.find_element_by_id("user_password")


    def _load_urls(self, url_of_pageofurls):
        self.driver.get(url_of_pageofurls)
        trails_limit = 0
        time.sleep(4)
        loadMoreButton = self.driver.find_element_by_xpath(
            "//button[contains(text(), 'Show more trails')]"
        )
        time.sleep(8)
        while trails_limit < 1001:
            try:
                loadMoreButton.click()
                time.sleep(10)
                trails_limit += 1
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


    def save_urls_to_json(self, url, filename_html, filename_json):
        if os.path.exists(filename):
            return
        else:
            self.get_urls(url, filename_html)

            with open(filename_json, 'w', encoding='utf-8') as f:
                json.dump(self.all_hike_urls, f, ensure_ascii=False, indent=4)
            

    def load_reviews(self, url):
        # while True:
        try:
            loadMoreButton = self.driver.find_element_by_xpath("//button[contains(text(), 'Show more reviews')]")
            loadMoreButton.click()
            time.sleep(5)
            # test
            html = self.driver.page_source
            return html
        except:
            html = self.driver.page_source
            return html

    def _make_soup(self, html):
        soup = BeautifulSoup(html, 'lxml')
        return soup

