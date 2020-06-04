import requests, time, re, json, pickle, os
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException, TimeoutException
from pymongo import MongoClient
import pandas as pd
import numpy as np 
from chromedriver_class import ChromeDriver
 
def scrape(driver_object, url_list, mongo_coll):
    '''Run scrape on all urls.'''
    if driver_object.all_hike_urls:
        urls = driver_object.all_hike_urls
    else:
        urls = url_list
    for url in urls:
        driver_object.driver.get(url)
        time.sleep(5)
        html = driver_object.load_reviews(url)
        soup = BeautifulSoup(html, 'lxml')
        hike_soup = get_hike_information(soup)
        mongo_coll.insert_one(hike_soup)
        time.sleep(4)
    return 
 
def get_hike_information(soup):
    '''
    Get all hike information from html.

    Input: Beautiful Soup
    Output: Dictionary
    '''
    description = get_hike_description(soup)
    secondary_description = get_second_description(soup)
    reviews = get_user_reviews(soup)
    hike_soup = soup.find('div', id='title-and-menu-box')
    try:
        hike_name = hike_soup.find('h1', itemprop="name").get_text()
    except:
        hike_name = None
    hike_difficulty = hike_difficulty_levels(hike_soup)  
    hike_type = hike_types(soup)
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

    trail_distance, trail_elevation = get_trail_stats(soup)

    tags = get_tags(soup)

    url = soup.find('div', id='main')['itemid']

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
    '''Get hike difficulty.'''
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
    '''Get hike type.'''
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
    '''Get trail distance and elevation gain.'''
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
    '''Get tags associated with the hike.'''
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
    '''Get main description of hike.'''
    hike_soup = soup.find('div', id='title-and-menu-box')
    info = soup.find('section', id='trail-top-overview-text')
    try:
        description = info.find('p', class_="xlate-google line-clamp-4").get_text()
    except:
        description = None
    return description

def get_second_description(soup):
    '''Get secondary description of hike, if applicable.'''
    hike_soup = soup.find('div', id="trail-detail-item") 
    try:
        secondary_description = hike_soup.get_text()
    except:
        secondary_description = None
    return secondary_description

def get_user_reviews(soup):
    '''Get user names, star ratings, text reviews.'''
    user_ratings = {}
    hike_soup = soup.find('div', id='title-and-menu-box')
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
    '''Json file to list for indexing.'''
    with open(filepath, "rb") as fp:   # Unjsoning 
        urls = json.load(fp)
    return urls

if __name__ == "__main__":
    mongo = True
    full_scrape = True

    chrome = ChromeDriver()
    urls = unjson('/Users/annierumbles/Desktop/Coding/galvanize/second_capstone_live/data/ALL_colorado_url_list.json')

    if mongo:
        client = MongoClient('localhost', 27017)
        db = client['hikes']
        colorado_hikes = db['colorado_hikes']

    if full_scrape:
        scrape(chrome, urls, colorado_hikes)

