#!/usr/bin/env python
# coding: utf-8

# In[62]:


from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import json


# 전체 페이지 수 가져오기
def get_page_num():
    url = 'https://www.myplate.gov/myplate-kitchen/recipes?f[0]=cuisine%3A139'
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')

    max_num = soup.find('nav', {'class': 'pager'}).contents[-2].contents[-2].contents[-2].get('href')[-2:]
    nums = list(range(1, int(max_num)))

    return nums


# 입력한 페이지의 전체 레시피 링크 가져오기
def get_links(i):
    link_list = list()
    url = 'https://www.myplate.gov/myplate-kitchen/recipes?f[0]=cuisine%3A139&page=' + str(i)
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    
    articles = soup.find_all('article', {'role': 'article'})
    
    for article in articles:
        link_list.append('https://www.myplate.gov' + article.contents[3].contents[1].contents[1].get('href'))
    
    return link_list


# 입력한 링크의 제목, 댓글(후기) 가져오기
def get_contents(url):
    contents = dict()
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    
    title = soup.select('h1 > span')[0].text
    comments = soup.find_all(class_='comment-body')[1:]
    
    com_list = list()
    for comment in comments:
        com_list.append(comment.text.strip())
    
    contents['title'] = title
    contents['comments'] = com_list
        
    return contents


# 전체 페이지 레시피 댓글 가져오기
def get_all_page_comment(nums):
    total = dict()
    title_comments = list()
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        for num in nums:
            links = get_links(num)
            for link in links:
                content = executor.submit(get_contents, link)
                title_comments.append(content.result())
    
    total['myplate'] = title_comments
    return total


# 메인에서 실행
if __name__ == '__main__':
    nums = get_page_num()
    total = get_all_page_comment(nums)
    
    with open('/home/ubuntu/crawling/raw_data/myplate_review_sample.json', 'w', encoding='utf-8-sig') as file:
        json.dump(total, file, indent="\t")

