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


# 입력한 링크의 내용 가져오기
def get_contents(url):
    contents = dict()
    soup = BeautifulSoup(requests.get(url).text, 'html.parser')
    
    # 출처 (필수)
    contents['site'] = url

    # 제목 (필수)
    try:
        title = soup.find('h1', {'class': 'entry-title'}).text
        contents['title'] = title
    except:
        print(url + '_error')

    # 재료 (필수)
    try:
        ing_div = soup.find('div', {'class': 'ingredients-section'})
        ing_list = list()

        for li in ing_div.find_all('li'):
            ing_list.append(li.text.strip().replace('*', ''))

        contents['ingredients'] = ing_list
    except:
        print(title + '_error_ingredients')
        

    # 조리시간
    try:
        contents['time'] = soup.find('div', {'class': 'cooking-time'}).find('div', {'class': 'value'}).text.strip()    
    except:
        pass

    # 분량
    try:
        unit = soup.find('div', {'class': 'yield'}).contents[1].contents[0].text
        num = soup.find('div', {'class': 'yield'}).contents[1].contents[1].text

        contents['serving'] = unit + ' ' + num
    except:
        pass

    # 레시피 (필수)
    try:
        instr = soup.find('div', {'class': 'method-section'})
        instr_list = list()

        if len(instr.find_all('p')) > 1:
            # 1번 소제목
            temp_dict = dict()
            instr_1 = instr.text.split('\n')[0].replace('METHOD', '')
            temp_list = list()

            lis1 = instr.find_all('p')[0].previous_sibling('li')

            for i in range(len(lis1)):
                temp_list.append(str(i+1) + ". " + lis1[i].text.strip())

            temp_dict[instr_1] = temp_list
            instr_list.append(temp_dict)

            # 2번 이후 소제목
            for p in instr.find_all('p'):
                if p.text != '':
                    temp = dict()
                    instr_temp = list()

                    lis2 = p.find_next_siblings()[0].contents

                    for i in range(len(lis2)):
                        if lis2[i] != '\n':
                            instr_temp.append(str(int((i+1)/2)) + ". " + lis2[i].text.strip())

                    temp[p.text] = instr_temp
                    instr_list.append(temp)

        else:
            lis = instr.find_all('li')
            for i in range(len(lis)):
                instr_list.append(str(i+1) + ". " + lis[i].text.strip())

        contents['recipe'] = instr_list
    except:
        print(title + '_error_recipe')

    # 영양정보
    try:
        nutri = soup.find('div', {'class': 'nutritional-info-boxes'})
        contents['calories'] = nutri.find('div', text='calories').find_next_sibling().contents[0].text.strip() + ' kcal'
        contents['carbs'] = nutri.find('div', text='carbs').find_next_sibling().contents[0].text.strip()
        contents['protein'] = nutri.find('div', text='proteins').find_next_sibling().contents[0].text.strip()
        contents['total_fat'] = nutri.find('div', text='fats').find_next_sibling().contents[0].text.strip()
  
    except:
        pass

    # 사진 (필수)
    try:
        contents['image'] = soup.find('div', {'class': 'post-content'}).img.get('src')
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
    
    total['lazycatkitchen'] = title_comments
    return total


# 메인에서 실행
if __name__ == '__main__':
    nums = get_page_num()
    total = get_all_page_comment(nums)
    
    with open('/home/ubuntu/crawling/raw_data/lazycatkitchen_review_all.json', 'w', encoding='utf-8-sig') as file:
        json.dump(total, file, indent="\t")
        
    print("done")