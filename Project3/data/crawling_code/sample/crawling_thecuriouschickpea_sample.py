from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import json
import re


# 전체 페이지 수 가져오기
def get_page_num():
    url = 'https://www.thecuriouschickpea.com/'
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')

    max_num = soup.find('div', {'class': 'nav-links'}).contents[-3].text
    nums = list(range(1, int(max_num)+1))

    return nums


# 입력한 페이지의 전체 레시피 링크 가져오기
def get_links(i):
    link_list = list()
    url = 'https://www.thecuriouschickpea.com/page/' + str(i)
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    
    titles = soup.find_all('h2', {'class': 'excerpt-title'})
    
    for title in titles:
        link_list.append(title.contents[0].get('href'))
    
    return link_list


# 입력한 링크의 제목, 댓글(후기) 가져오기
def get_contents(url):
    contents = dict()
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    
    title = soup.select('h1')[0].text
    
    page_num = soup.find('link', {'rel': 'shortlink'}).get('href')[-4:]
    comments_url = 'https://www.thecuriouschickpea.com/wp-json/wp/v2/comments?post=' + page_num +'&per_page=100'
    comments = requests.get(comments_url).json()
    
    com_list = list()
    for comment in comments:
        try:
            if (comment['author_name'] != 'thecuriouschickpea') and (comment['author_name'] != 'Eva Agha'):
                com_list.append(re.sub('(<([^>]+)>)', '', comment['content']['rendered']).strip())
        except:
            pass

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
    
    total['thecuriouschickpea'] = title_comments
    return total


# 메인에서 실행
if __name__ == '__main__':
    nums = get_page_num()
    total = get_all_page_comment(nums)

    with open('/home/ubuntu/crawling/raw_data/thecuriouschickpea_review_sample.json', 'w', encoding='utf-8-sig') as file:
        json.dump(total, file, indent="\t")