from concurrent.futures import ThreadPoolExecutor
import requests
from bs4 import BeautifulSoup
import json
import re


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


# 입력한 링크의 내용 가져오기
def get_contents(url):
    contents = dict()
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    
    # 출처 (필수)
    contents['site'] = url

    # 제목 (필수)
    try:
        title = soup.select('h1 > span')[0].text
        contents['title'] = title
    except:
        print(url + '_error')

    # 재료 (필수)
    try:
        ing_div = soup.find('div', {'class': 'field--name-field-ingredients'})
        ing_list = list()

        for li in ing_div.find_all('li'):
            proc_str = ''
            for word in list(filter(None, re.split('\s', li.text))):
                proc_str = proc_str + ' ' + word
            ing_list.append(proc_str.strip())

        contents['ingredients'] = ing_list
    except:
        print(title + '_error_ingredients')

    # 조리시간
    try:
        contents['time'] = soup.find('div', {'class': 'mp-recipe-full__detail--prep-time'}).find_all('span')[-1].text.strip()
    except:
        pass

    # 분량
    try:
        serv = soup.find('div', {'class': 'mp-recipe-full__detail--yield'}).find_all('span')[-1].text.strip()
        contents['serving'] = re.sub('\n| ', '', serv).replace('Serv', ' Serv').replace('serv', ' serv')
    except:
        pass

    # 레시피 (필수)
    try:
        instr_list = list()        
        instr_div = soup.find(class_='field--name-field-instructions').find('div', {'class': 'field__item'})

        lis = instr_div.find_all('li')
        ps = instr_div.find_all('p')

        if len(lis) == 0:
            for i in range(len(ps)):
                instr_list.append(ps[i].text.strip())
            contents['recipe'] = instr_list
        elif len(lis) == 1:
            contents['recipe'] = lis[0].text.strip()
        else:
            for i in range(len(lis)):
                instr_list.append(str(i+1) + ". " + lis[i].text.strip())
            contents['recipe'] = instr_list
    except:
        print(title + '_error_recipe')

    # 영양정보
    try:
        contents['calories'] = soup.find('tr', {'class': 'total_calories'}).find_all('td')[-1].text.strip() + 'kcal'
        contents['carbs'] = soup.find('tr', {'class': 'carbohydrates'}).find_all('td')[-1].text.strip().replace(' ', '')
        contents['protein'] = soup.find('tr', {'class': 'protein'}).find_all('td')[-1].text.strip().replace(' ', '')
        contents['total_fat'] = soup.find('tr', {'class': 'total_fat'}).find_all('td')[-1].text.strip().replace(' ', '')

    except:
        pass

    # 사진 (필수)
    try:
        contents['image'] = soup.find(class_='field--name-field-recipe-image').find('div', {'class': 'field__item'}).img.get('src')
    except:
        print(title + '_error_image')

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
    
    with open('/home/ubuntu/crawling/raw_data/myplate_review_all.json', 'w', encoding='utf-8-sig') as file:
        json.dump(total, file, indent="\t")
        
    print("done")