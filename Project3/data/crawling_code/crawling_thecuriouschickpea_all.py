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


# 입력한 링크의 내용 가져오기
def get_contents(url):
    contents = dict()
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    
    # 출처 (필수)
    contents['site'] = url

    # 제목 (필수)
    try:
        title = soup.select('h1')[0].text
        contents['title'] = title
    except:
        print(url + '_error')

    # 재료 (필수)
    try:
        ing_div = soup.find('div', {'class': 'mv-create-ingredients'}).find_all('li')
        ing_list = list()

        for li in ing_div:
            ing_list.append(li.text.strip().replace('*', ''))

        contents['ingredients'] = ing_list
    except:
        print(title + '_error_ingredients')

    # 조리시간
    try:
        contents['time'] = soup.find('div', {'class': 'mv-create-time mv-create-time-total'}).find('span').text.strip()
    except:
        pass

    # 분량
    try:
        contents['serving'] = soup.find('div', {'class': 'mv-create-time mv-create-time-yield'}).find('span').text.strip()
    except:
        pass

    # 레시피 (필수)
    try:
        instr = soup.find('div', {'class': 'mv-create-instructions'}).find_all('li')
        instr_list = list()

        for i in range(len(instr)):
            instr_list.append(str(i+1) + ". " + instr[i].text)

        contents['recipe'] = instr_list
    except:
        print(title + '_error_recipe')

    # 영양정보
    try:
        nutri = soup.find('div', {'class': 'mv-create-nutrition-box'})

        contents['calories'] = nutri.find('span', {'class': 'mv-create-nutrition-calories'}).text.replace('Calories: ','') + 'kcal'
        contents['carbs'] = nutri.find('span', {'class': 'mv-create-nutrition-carbohydrates'}).text.replace('Carbohydrates: ','')
        contents['protein'] = nutri.find('span', {'class': 'mv-create-nutrition-protein'}).text.replace('Protein: ','')
        contents['total_fat'] = nutri.find('span', {'class': 'mv-create-nutrition-total-fat'}).text.replace('Total Fat: ','')

    except:
        pass

    # 사진 (필수)
    try:
        image_div = soup.find('div', {'class': 'wp-block-image'})
        image_figure = soup.find('figure', {'class': 'wp-block-image'})
        image_entry = soup.find('div', {'class': 'entry-content mvt-content'})

        if image_div is not None:
            contents['image'] = image_div.img.get('src')
        elif image_figure is not None:
            contents['image'] = image_figure.img.get('src')
        elif image_entry is not None:
            contents['image'] = image_entry.img.get('src')
        else:
            print(title + '_error_image')
    except:
        print(title + '_error_image')

    return contents


# 전체 페이지 레시피 가져오기
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

    with open('/home/ubuntu/crawling/raw_data/thecuriouschickpea_review_all.json', 'w', encoding='utf-8-sig') as file:
        json.dump(total, file, indent="\t")
        
    print("done")