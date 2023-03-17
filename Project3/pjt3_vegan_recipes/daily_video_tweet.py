from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from time import sleep
import random
from pymongo import MongoClient
from datetime import datetime, timezone
from dateutil import parser
import requests
from .BASE_DIR import *


def today_yt():
    url = 'https://www.youtube.com/results?search_query=vegan+recipe&sp=CAMSBAgCEAE%253D'

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    service = Service(BASE_DIR + CHROMEDRIVER)
    driver = webdriver.Chrome(service=service, options=options)
    driver.get(url)
    sleep(3)

    v_list = list()

    for i in range(10):
        v_path = '/html/body/ytd-app/div[1]/ytd-page-manager/ytd-search/div[1]/ytd-two-column-search-results-renderer/div/ytd-section-list-renderer/div[2]/ytd-item-section-renderer/div[3]/ytd-video-renderer[' + str(i + 1) + ']/div[1]/ytd-thumbnail/a'
        v_list.append(driver.find_element(By.XPATH, v_path).get_attribute("href"))

    ran_vid = random.choice(v_list)

    if "shorts" not in ran_vid:
        today_vid = ran_vid.replace('/watch?v=', '/embed/')
    else:
        today_vid = ran_vid.replace('shorts', 'embed')

    return today_vid


def today_tw():
    # client = MongoClient('localhost', 27017)
    client = MongoClient('35.79.107.247', 27017)
    db = client.test

    tweets = db.twitter.find({}, {'_id': 0})
    t_list = list()

    today = datetime.now(timezone.utc)

    for tweet in tweets:
        author_id = tweet['author_id']
        tweet_id = tweet['id']
        link = f"https://twitter.com/{author_id}/status/{tweet_id}"

        # 날짜가 오늘인 트윗만 출력
        tweet_date = tweet['created_at']
        tweet_date_parse = parser.parse(tweet_date)

        if (today - tweet_date_parse).seconds / 3600 <= 24:
            t_list.append(link)

    rand_twt = random.choice(t_list)

    embed_query = "https://publish.twitter.com/oembed?url="
    req_json = requests.get(embed_query + rand_twt).json()

    response_text = req_json['html']

    return response_text
