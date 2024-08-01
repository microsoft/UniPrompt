import json
import pandas as pd


#Copyright (c) Microsoft Corporation. All rights reserved.
#Licensed under the MIT License.

# -*- coding: utf-8 -*-

import json
import os 
from pprint import pprint
import requests

'''
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = 'c37f4952d17c4f518dcc25a0e0711ff7'
endpoint = 'https://api.bing.microsoft.com/v7.0/search'

# Query term(s) to search for. 
# query = "wolf lodge"
def make_query(query, mkt):
    # Construct a request
    # mkt = 'en-US'
    # mkt = 'global'
    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }
    # breakpoint()
    # Call the API
    Flag = False
    while(not Flag):
        try:
            response = requests.get(endpoint, headers=headers, params=params)
            response.raise_for_status()
            x = response.json()
            resp1 = x['webPages']['value'][0]['name']
            resp2 = x['webPages']['value'][1]['name']
            resp3 = x['webPages']['value'][2]['name']
            resp4 = x['webPages']['value'][3]['name']
            resp5 = x['webPages']['value'][4]['name']
            resp6 = x['webPages']['value'][5]['name']
            resp7 = x['webPages']['value'][6]['name']
            resp8 = x['webPages']['value'][7]['name']
            resp9 = x['webPages']['value'][8]['name']
            resp10 = x['webPages']['value'][9]['name']
            Flag = True
            
        except:
            pass
    return resp1, resp2, resp3, resp4, resp5, resp6, resp7, resp8, resp9, resp10

# read from the file Bing_Ads/retrivals.jsonl that has format "query", "keyword"
# Replace the file path with the actual file path in line 61
# replace column names with the actual column names in line 66, 72, 73, 74
def read_bing_ads(Languages, Country):
    final_data = []
    file_path = 'En-gb.txt'
    data = pd.read_csv(file_path, sep='\t')
    with open(file_path, 'r', errors='ignore', encoding = 'utf-8') as file:
        lines = file.readlines()
    data = [line.strip().split('\t') for line in lines]
    columns = ['GPNQuery', 'GPNKeyword', 'GroundTruth']
    data = pd.DataFrame(data, columns=columns)
    # print(data.head())
    breakpoint()

    # Display the contents of the DataFramedata
    Query = data['GPNQuery']
    Keyword = data['GPNKeyword']
    Ground_truth = data['GroundTruth']
    for i in range(len(Query)):
            query = Query[i]
            keyword = Keyword[i]
            ground_truth = Ground_truth[i]
        


            querytitle1, querytitle2, querytitle3, querytitle4, querytitle5, querytitle6, querytitle7, querytitle8, querytitle9, querytitle10 = make_query(query, "en-US")
            keywordtitle1, keywordtitle2, keywordtitle3, keywordtitle4, keywordtitle5, keywordtitle6, keywordtitle7, keywordtitle8, keywordtitle9, keywordtitle10 = make_query(keyword, "en-US")

            final_string = f'''For below query A and query B both in {Languages} language from a user based in {Country}, as well as their search result title :
            query A: {query} 
            query A's first search result title : {querytitle1}
            query A's second search result title : {querytitle2}
            query A's third search result title : {querytitle3}
            query B: {keyword}
            query B's first search result title: {keywordtitle1}
            query B's second search result title: {keywordtitle2}
            query B's third search result title: {keywordtitle3}
            Tell if they have same intent or not.
            '''
            final_data.append([final_string, ground_truth, ["No", "Yes"]])
    return final_data

Languages = "English"
Country = "United States"
final_dataset = read_bing_ads(Languages, Country)

# write to the file EEM.jsonl in hte format "split":, "question": , "choices":, "answer": 
# Give first 200 as training data and next 50 as validation data rest as test data
# replace the file path with the actual file path in line 108
with open("eem_three_retrivals.jsonl", "w") as f:
    for i, data in enumerate(final_dataset):
        split = "test"
        f.write(json.dumps({"split": split, "question": data[0], "choices": data[2], "answer": data[1]}) + "\n")

