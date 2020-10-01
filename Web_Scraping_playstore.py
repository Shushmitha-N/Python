# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 12:15:05 2020

@author: shushmitha natarajan
"""

from google_play_scraper import app

result = app(
    'com.activision.callofduty.shooter',
    lang='en', # defaults to 'en'
    country='us' # defaults to 'us'
)

# from google_play_scraper import Sort, reviews
"""
result, continuation_token = reviews(
    'com.activision.callofduty.shooter',
    lang='en', # defaults to 'en'
    country='us', # defaults to 'us'
    sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT
    count=3, # defaults to 100
    filter_score_with=5 # defaults to None(means all score)
)

# If you pass `continuation_token` as an argument to the reviews function at this point,
# it will crawl the items after 3 review items.

result, _ = reviews(
    'com.activision.callofduty.shooter',
    continuation_token=continuation_token # defaults to None(load from the beginning)
)
"""
from google_play_scraper import Sort, reviews_all

result = reviews_all(
    'com.activision.callofduty.shooter',
    sleep_milliseconds=0, # defaults to 0
    lang='en', # defaults to 'en'
    country='us', # defaults to 'us'
    sort=Sort.MOST_RELEVANT, # defaults to Sort.MOST_RELEVANT
    filter_score_with= None # defaults to None(means all score)
)

print(result[1]["userName"])

y = []
for i in range(len(result)):
    a = result[i]   
    info = {'Name': a["userName"],'review':a['content'],'score':a['score']}
    y.append(info)
    i = i+1

import csv

toCSV = y
with open('Reviews_stash.csv', 'w', encoding='utf8', newline='') as output_file:
    fc = csv.DictWriter(output_file, 
                        fieldnames=toCSV[0].keys(),
                       
                       )
    fc.writeheader()
    fc.writerows(toCSV)
