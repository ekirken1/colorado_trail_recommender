import requests, time, re, json, pickle, os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from pymongo import MongoClient
import pandas as pd
import numpy as np 
from src.chromedriver_class import ChromeDriver
from src.configs.get_configs import get_all_configs
from src.data_connections.dbs import get_mongo_data


def scrape(driver_object, url_list, mongo_coll):
    '''Scrapes and cleans reviews for each trail url.
    
        Args:
            driver_object (ChromeDriver object):
                connection to website
            url_list (JSON): urls in JSON format
            mongo_coll (MongoDB cluster): Mongo db to store data
    '''
    if driver_object.all_hike_urls:
        urls = driver_object.all_hike_urls
    else:
        urls = url_list

    for url in urls:
        driver_object.driver.get(url)
        time.sleep(5)
        html = driver_object.load_reviews(url)
        soup = BeautifulSoup(html, 'lxml')
        hike_soup = get_hike_information(soup, mongo_coll)
        mongo_coll.insert_one(hike_soup)
        time.sleep(4)

    print(f"Number of hikes: {mongo_coll.count_documents({})}")

    return
 

def get_hike_information(soup, hikes):
    '''Get all hike information from html.

        Args:
            soup (BeautifulSoup object): parsed HTML data
            hikes (MongoDB cluster): Mongo db to store data

        Returns:
            hike (dict): hike data
    '''
    description = get_hike_description(soup)
    secondary_description = get_second_description(soup)
    reviews = get_user_reviews(soup)
    hike_soup = soup.find('div', id='title-and-menu-box')

    try:
        hike_name = hike_soup.find('h1', itemprop="name").get_text()
    except:
        hike_name = None

    if hikes.find_one({"hike_name": hike_name}):
        return

    try:
        hike_difficulty = hike_difficulty_levels(hike_soup)
    except:
        hike_difficulty = None

    try:
        hike_type = hike_types(soup)
    except:
        hike_type = None

    try:
        rating = float(hike_soup.find('meta', itemprop='ratingValue')['content'])
    except:
        rating = None

    try:
        rating_count = int(hike_soup.find('meta', itemprop='reviewCount')['content'])
    except:
        rating_count = None

    try:
        general_location = hike_soup.find('a', class_="xlate-none styles-module__location___11FHK styles-module__location___3wEnO").get_text()
    except:
        general_location = None

    try:
        trail_distance, trail_elevation = get_trail_stats(soup)
    except:
        trail_distance, trail_elevation = None

    try:
        tags = get_tags(soup)
    except:
        tags = None

    try:
        url = soup.find('div', id='main')['itemid']
    except:
        url = None

    hike = {'name': hike_name,
            'url': url,
            'difficulty': hike_difficulty,
            'hike_type': hike_type,
            'avg_rating': rating,
            'number_ratings': rating_count,
            'location': general_location,
            'distance': trail_distance,
            'elevation': trail_elevation,
            'tags': tags,
            'main_description': description,
            'secondary_description': secondary_description,
            'reviews': reviews}

    return hike


def hike_difficulty_levels(hike_soup):
    """Parse hike difficulty from HTML.

        Args:
            hike_soup (BeautifulSoup object): parsed HTML data

        Returns:
            hike_difficulty (str): difficulty rating of hike
    """
    if hike_soup.find('span', class_="styles-module__diff___22Qtv styles-module__hard___3zHLb styles-module__selected___3fawg"):
        hike_difficulty = 'hard'
    elif hike_soup.find('span', class_="styles-module__diff___22Qtv styles-module__easy___bPX-K styles-module__selected___3fawg"):
        hike_difficulty = 'easy'
    elif hike_soup.find('span', class_="styles-module__diff___22Qtv styles-module__moderate___3w1it styles-module__selected___3fawg"):
        hike_difficulty = 'moderate'
    else:
        hike_difficulty = None

    return hike_difficulty


def hike_types(soup):
    """Parse type of hike from HTML.
    
        Args:
            soup (BeautifulSoup object): parsed HTML data

        Returns:
            hike_type (str): type of hike
    """
    trail_stats = soup.find('section', id="trail-stats")
    if trail_stats.find('span', class_="route-icon out-and-back"):
        hike_type = 'Out & Back'
    elif trail_stats.find('span', class_="route-icon loop"):
        hike_type = 'Loop'
    elif trail_stats.find('span', class_="route-icon point-to-point"):
        hike_type = "Point to Point"
    else:
        hike_type = None

    return hike_type


def get_trail_stats(soup):
    """Parse trail distance and elevation from HTML.

        Args:
            soup (BeautifulSoup object): parsed HTML data

        Returns:
            trail_distance (float or int): distance of trail in miles
            trail_elevation (float or int): trail elevation in feet
    """
    trail_stats = soup.find('section', id="trail-stats")
    try:
        trail_distance = trail_stats.find('span', class_='distance-icon').find('span', class_="detail-data xlate-none").get_text()
        if 'km' in trail_distance:
            try:
                trail_distance = float(re.findall("\d+\.\d+", trail_distance)[0])
                trail_distance = round((trail_distance/1.609), 1)
            except:
                trail_distance = [int(s) for s in trail_distance.split() if s.isdigit()][0]
                trail_distance = round((trail_distance/1.609), 1)
        else:
            try:
                trail_distance = float(re.findall("\d+\.\d+", trail_distance)[0])
            except:
                trail_distance = [int(s) for s in trail_distance.split() if s.isdigit()][0]
    except:
        trail_distance = None

    try:
        trail_elevation = trail_stats.find('span', class_='elevation-icon').find('span', class_="detail-data xlate-none").get_text()
        if 'm' in trail_elevation:
            try:
                trail_elevation = int(re.findall("\d+\,\d+", trail_elevation)[0].replace(",", ""))
                trail_elevation = round((trail_elevation*3.28084),1)
            except:
                trail_elevation = [int(s) for s in trail_elevation.split() if s.isdigit()][0]
                trail_elevation = round((trail_elevation*3.28084),1)
        else:
            try:
                trail_elevation = int(re.findall("\d+\,\d+", trail_elevation)[0].replace(",", ""))
            except:
                trail_elevation = [int(s) for s in trail_elevation.split() if s.isdigit()][0] 
    except:
        trail_elevation = None
    return trail_distance, trail_elevation


def get_tags(soup):
    """Get tags associated with each trail.

        Args:
            soup (BeautifulSoup object): parsed HTML data

        Returns:
            tag_string(str): string of tags of each trail
    """
    tags = soup.find('section', class_="tag-cloud") 
    tag_object_list = [tag for tag in list(tags.children)] 
    tag_string = ''
    for i in range(len(tag_object_list)):
        if repr(type((tag_object_list[i]))) == "<class 'bs4.element.Tag'>":
            tag = tag_object_list[i].find('h3').find('span', class_="big rounded active").get_text()
            tag_string += (tag+' ')
        else:
            continue
    return tag_string


def get_hike_description(soup):
    """Get main description of trail.

        Args:
            soup (BeautifulSoup object): parsed HTML data

        Returns:
            description (str): description of trail
    """
    # hike_soup = soup.find('div', id='title-and-menu-box')
    info = soup.find('section', id='trail-top-overview-text')

    try:
        description = info.find('p', class_="xlate-google line-clamp-4").get_text()
    except:
        description = None

    return description


def get_second_description(soup):
    """Get secondary description of hike, if applicable.

        Args:
            soup (BeautifulSoup object): parsed HTML data

        Returns:
            secondary_description (str): second description of trail
    """
    hike_soup = soup.find('div', id="trail-detail-item")
 
    try:
        secondary_description = hike_soup.get_text()
    except:
        secondary_description = None

    return secondary_description


def get_user_reviews(soup):
    """Parse ratings by username.
    
        Args:
            soup (BeautifulSoup object): parsed HTML data

        Returns:
            user_ratings (dict): ratings by username.
    """
    user_ratings = {}
    # hike_soup = soup.find('div', id='title-and-menu-box')

    try:
        reviews = soup.find('div', class_='feed-items')
        review_list = [rev for rev in list(reviews.children)]
        for rev in review_list:
            user_name = rev.find('span', itemprop='author').get_text().lower().replace(' ', '_').replace('.', '')
            if user_name == '$15_$40':
                user_name = 'dollar15_dollar40'
            user_rating = int(rev.find('meta', itemprop="ratingValue")['content'])
            user_review = rev.find('p', itemprop='reviewBody').get_text()
            try:
                tags = rev.find('span', class_='review-tags')
                tag_list = [t for t in list(tags.children)]
                tag_string = ''
                for t in tag_list:
                    tag_string += (t.get_text()+' ')
            except:
                tag_string = None
            user_ratings[user_name] = user_rating, tag_string, user_review
    except:
        user_ratings = None
        return user_ratings

    return user_ratings


def unjson(filepath):
    """Load and unpack json from file.
    
        Args:
            filepath (str): path to json file
        
        Returns:
            urls (): trails and their urls from json file
    """
    with open(filepath, "rb") as fp:   # Unjsoning 
        urls = json.load(fp)
    return urls


def main():
    config_file = "src/configs/config.ini"
    html_file_path = "data/html.txt"
    json_file_path = "data/alltrails_colorado_hikes.json"
    colorado_trails_url = "https://www.alltrails.com/us/colorado"
    chrome = ChromeDriver()
    configs = get_all_configs(config_file)
    mongo_configs = configs["mongo"]

    client, colorado_hikes = get_mongo_data(mongo_configs)

    if colorado_hikes.count_documents({}) > 0:
        if os.path.exists(json_file_path):
            print("Unpacking trail urls...")
            urls = unjson(json_file_path)
            urls.sort()
        else:
            print("Scraping trail urls...")
            chrome.save_urls_to_json(colorado_trails_url, html_file_path, json_file_path)
            
        print("Scraping all urls...")
        scrape(chrome, urls, colorado_hikes)

    return client, colorado_hikes


if __name__ == "__main__":
    client, colorado_hikes = main()

