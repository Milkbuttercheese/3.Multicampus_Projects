from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import json


# 전체 페이지 수 가져오기
def get_page_num():
    url = 'https://www.lazycatkitchen.com/category/recipes/'
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')

    max_num = soup.find('div', {'class': 'archive-pagination'}).contents[-3].text
    nums = list(range(1, int(max_num) + 1))

    return nums


# 입력한 페이지의 전체 레시피 링크 가져오기
def get_links(i):
    link_list = list()
    url = 'https://www.lazycatkitchen.com/category/recipes/page/' + str(i)
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    articles = soup.find_all('article', {'class': 'post-summary'})
    
    for article in articles:
        link_list.append(article.contents[1].get('href'))
    
    return link_list


# 입력한 링크의 제목, 댓글(후기) 가져오기
def get_contents(url):
    contents = dict()
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    
    title = soup.find('h1', {'class': 'entry-title'}).text
    comments = soup.find_all('div', {'class': 'comment-body-inner not-ania'})
    
    com_list = list()
    for comment in comments:
        com_list.append(comment.contents[-1].text)
    
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
    
    total['lazycatkitchen'] = title_comments
    return total


# 메인에서 실행
if __name__ == '__main__':
    nums = get_page_num()
    total = get_all_page_comment(nums)
    
    with open('/home/ubuntu/crawling/raw_data/lazycatkitchen_review_sample.json', 'w', encoding='utf-8-sig') as file:
        json.dump(total, file, indent="\t")

