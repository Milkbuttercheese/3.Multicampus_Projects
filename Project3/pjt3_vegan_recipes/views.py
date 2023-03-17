from django.shortcuts import render, redirect
from django.core.paginator import Paginator, EmptyPage, InvalidPage
from django.db.models import Q
from datetime import timedelta

from .recommender_systems import *
from .daily_video_tweet import *
from .models import *
from .BASE_DIR import BASE_DIR

import sys
sys.path.append(BASE_DIR)
import json


# 로그인 전 메인
def main(request):
    category_region = dict()
    # category
    category_1_total = Recipe.objects.filter(category='1.India+South America+South Asia <Main ingredients: cumin/coriander/cilantro/lime/avocado/onion>')
    category_1_id_list = list()
    for data in category_1_total:
        category_1_id_list.append(data.recipe_id)
    c1_len = len(category_1_id_list)
    c1_id = random.choice(category_1_id_list)
    category_1 = Recipe.objects.get(recipe_id=c1_id)
    category_region['1'] = '1. India + South America + South Asia'

    category_2_total = Recipe.objects.filter(category='2.East Asia <Main ingredients: rice/soy/sesame/tofu>')
    category_2_id_list = list()
    for data in category_2_total:
        category_2_id_list.append(data.recipe_id)
    c2_len = len(category_2_id_list)
    c2_id = random.choice(category_2_id_list)
    category_2 = Recipe.objects.get(recipe_id=c2_id)
    category_region['2'] = '2. East Asia'

    category_3_total = Recipe.objects.filter(category='3.Dessert+ Bread <Main ingredients: sugar/milk/coconut/vanilla/butter/almond>')
    category_3_id_list = list()
    for data in category_3_total:
        category_3_id_list.append(data.recipe_id)
    c3_len = len(category_3_id_list)
    c3_id = random.choice(category_3_id_list)
    category_3 = Recipe.objects.get(recipe_id=c3_id)
    category_region['3'] = '3. Dessert + Bread'

    category_4_total = Recipe.objects.filter(category='4.West+Etc')
    category_4_id_list = list()
    for data in category_4_total:
        category_4_id_list.append(data.recipe_id)
    c4_len = len(category_4_id_list)
    c4_id = random.choice(category_4_id_list)
    category_4 = Recipe.objects.get(recipe_id=c4_id)
    category_region['4'] = '4. West + Etc'

    # youtube
    today_video = today_yt()

    # twitter
    # today_twitter = today_tw()
    # 'today_tw': today_twitter

    # 통계 섹션
    counts = dict()

    # 통계 섹션 - 레시피 수
    recipeDF = pd.DataFrame(list(Recipe.objects.all().values()))
    counts['recipe_count'] = recipeDF['recipe_id'].count()

    # 통계 섹션 - 유저 수
    user_all = UserInfo.objects.all()
    counts['user_count'] = len(user_all)

    # 통계 섹션 - 찜 레시피 수
    pinned_recipe_all = PinnedRecipe.objects.all()
    counts['pinned_recipe_count'] = len(pinned_recipe_all)

    return render(request, 'main.html', {'category_1': category_1, 'category_2': category_2, 'category_3': category_3,
                                         'category_4': category_4, 'category_region': category_region,
                                         'today_yt': today_video, 'counts': counts})


def signup_info(request):
    return render(request, 'signup_info.html')


def signup_recipe(request):
    return render(request, 'signup_recipe.html')


def main_login(request):
    # user = request.session['user']
    # print(user)
    return render(request, 'main_login.html')


# 로그인
def login(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    elif request.method == 'POST':
        user_name = request.POST.get('user_name', None)
        user_pw = request.POST.get('user_pw', None)

        err_data = {}
        if not (user_name and user_pw):
            err_data['error'] = 'Please enter all fields'
            return render(request, 'login.html', err_data)
        else:
            user = UserInfo.objects.get(user_name=user_name)

            if user_pw == user.user_pw:
                request.session['user'] = user.user_id
                return redirect('/main_login')
            else:
                err_data['error'] = 'Wrong User_id or Password. Please Try Again.'
                return render(request, 'login.html', err_data)


# 로그아웃
def logout(request):
    if request.session.get('user'):
        del (request.session['user'])
    return redirect('/')


def recipe(request, id):
    recipe_one = Recipe.objects.get(recipe_id=id)
    user = request.session['user']
    rated_stars = Rating.objects.filter(user_id=user).filter(recipe_id=id)

    pin_recipe = dict()
    pinned = PinnedRecipe.objects.filter(user_id=user).filter(recipe_id=id)
    pinned_length = len(pinned)

    pin_recipe['pinned'] = pinned
    pin_recipe['pinned_length'] = pinned_length

    # 재료 덩어리 리스트로 만들기 #
    ingredients = recipe_one.ingredients
    ingredients = ingredients.split('[')[1]
    ingredients = ingredients.split(']')[0]
    ingredient_list = list()
    ingredient_list = ingredients.split(',')

    # 레시피 덩어리 리스트로 만들기 #
    recipe_bulk = recipe_one.recipe
    recipe_bulk = recipe_bulk.split('[')[1]
    recipe_bulk = recipe_bulk.split(']')[0]
    recipe_tmplist = list()
    # 레시피 방법 순으로 자르기 ('"' 기준으로 구분 > 홀수 요소만 추출)
    recipe_tmplist = recipe_bulk.split('"')
    recipe_tmplist = recipe_tmplist[1::2]
    # 숫자 표시 지우고 리스트에 담기
    recipe_list = list()
    for recipe_item in recipe_tmplist:
        point = recipe_item.index('.')
        recipe_item = recipe_item[(point + 2):]
        recipe_list.append(recipe_item)

    category_raw = recipe_one.category
    category_index = category_raw.find('<')
    if category_index != -1:
        category_region = category_raw[:category_index]
    else:
        category_region = category_raw

    return render(request, 'recipe.html',
                  {'list': recipe_one, 'ingredient_list': ingredient_list, 'recipe_list': recipe_list,
                   'category_region': category_region, 'rated_stars': rated_stars, 'pin_recipe': pin_recipe})


def rate(request, id):
    recipe_one = Recipe.objects.get(recipe_id=id)
    user = request.session['user']
    stars = request.POST.get('ratingRadioOptions', None)
    print(user)
    print(stars)
    rating = Rating(user_id=user, recipe_id=id, selected_recipe_name=recipe_one.title, stars=stars)
    rated_stars = Rating.objects.filter(user_id=user).filter(recipe_id=id)
    print(rated_stars)
    if rated_stars != '<QuerySet []>':
        rating.save()
    else:
        pass
    return redirect('/recipe/' + str(id))


def pin_recipe(request, id):
    recipe_one = Recipe.objects.get(recipe_id=id)
    user = request.session['user']
    date = datetime.today().strftime("%Y-%m-%d")
    pinning = PinnedRecipe(
        user_id=user,
        recipe=recipe_one,
        date=date
    )
    pinning_recipe = PinnedRecipe.objects.filter(user_id=user).filter(recipe_id=id)
    pinning_recipe_length = len(pinning_recipe)

    if pinning_recipe_length == 1:
        pass
    else:
        pinning.save()

    return redirect('/recipe/' + str(id))


def signup_1(request):
    if request.method == 'GET':
        return render(request, 'signup_1.html')
    elif request.method == 'POST':
        user_name = request.POST.get('user_name', None)
        user_pw = request.POST.get('user_pw', None)
        re_user_pw = request.POST.get('re_user_pw', None)

        err_data = {}
        if not (user_name and user_pw and re_user_pw):
            err_data['error'] = 'Please enter all fields'
            return render(request, 'signup_1.html', err_data)
        elif user_pw != re_user_pw:
            err_data['error'] = 'Please check the password'
            return render(request, 'signup_1.html', err_data)
        else:
            user = UserInfo(user_name=user_name, user_pw=user_pw,)
            user.save()
            request.session['user_name'] = user_name

            return redirect('/signup_2/')


def signup_2(request):
    # category
    category_1_total = Recipe.objects.filter(category='1.India+South America+South Asia <Main ingredients: cumin/coriander/cilantro/lime/avocado/onion>')
    category_1_id_list = list()
    for data in category_1_total:
        category_1_id_list.append(data.recipe_id)
    c1_len = len(category_1_id_list)
    c1_id = random.choice(category_1_id_list)
    category_1 = Recipe.objects.get(recipe_id=c1_id)
    category_region = dict()
    category_region['1'] = '1. India + South America + South Asia'

    return render(request, 'signup_2.html', {'category_1': category_1, 'category_region': category_region})


def signup_3(request):
    # category
    category_2_total = Recipe.objects.filter(category='2.East Asia <Main ingredients: rice/soy/sesame/tofu>')
    category_2_id_list = list()
    for data in category_2_total:
        category_2_id_list.append(data.recipe_id)
    c2_len = len(category_2_id_list)
    c2_id = random.choice(category_2_id_list)
    category_2 = Recipe.objects.get(recipe_id=c2_id)
    category_region = dict()
    category_region['2'] = '2. East Asia'

    return render(request, 'signup_3.html', {'category_2': category_2, 'category_region': category_region})


def signup_4(request):
    # category
    category_3_total = Recipe.objects.filter(category='3.Dessert+ Bread <Main ingredients: sugar/milk/coconut/vanilla/butter/almond>')
    category_3_id_list = list()
    for data in category_3_total:
        category_3_id_list.append(data.recipe_id)
    c3_len = len(category_3_id_list)
    c3_id = random.choice(category_3_id_list)
    category_3 = Recipe.objects.get(recipe_id=c3_id)
    category_region = dict()
    category_region['3'] = '3. Dessert + Bread'

    return render(request, 'signup_4.html', {'category_3': category_3, 'category_region': category_region})


def signup_5(request):
    # category
    category_4_total = Recipe.objects.filter(category='4.West+Etc')
    category_4_id_list = list()
    for data in category_4_total:
        category_4_id_list.append(data.recipe_id)
    c4_len = len(category_4_id_list)
    c4_id = random.choice(category_4_id_list)
    category_4 = Recipe.objects.get(recipe_id=c4_id)
    category_region = dict()
    category_region['4'] = '4. West + Etc'

    return render(request, 'signup_5.html', {'category_4': category_4, 'category_region': category_region})


def signup_rate_1(request, id):
    recipe_one = Recipe.objects.get(recipe_id=id)
    user_name = request.session['user_name']
    user_one = UserInfo.objects.get(user_name=user_name)
    user = user_one.user_id
    stars = request.POST.get('ratingRadioOptions', None)
    # print(stars)
    rating = Rating(user_id=user, recipe_id=id, selected_recipe_name=recipe_one.title, stars=stars)
    rated_stars = Rating.objects.filter(user_id=user).filter(recipe_id=id)
    # print(rated_stars)
    if rated_stars != '<QuerySet []>':
        rating.save()
    else:
        pass
    return redirect('/signup_3/')


def signup_rate_2(request, id):
    recipe_one = Recipe.objects.get(recipe_id=id)
    user_name = request.session['user_name']
    user_one = UserInfo.objects.get(user_name=user_name)
    user = user_one.user_id
    stars = request.POST.get('ratingRadioOptions', None)
    print(user)
    print(stars)
    rating = Rating(user_id=user, recipe_id=id, selected_recipe_name=recipe_one.title, stars=stars)
    rated_stars = Rating.objects.filter(user_id=user).filter(recipe_id=id)
    print(rated_stars)
    if rated_stars != '<QuerySet []>':
        rating.save()
    else:
        pass
    return redirect('/signup_4/')


def signup_rate_3(request, id):
    recipe_one = Recipe.objects.get(recipe_id=id)
    user_name = request.session['user_name']
    user_one = UserInfo.objects.get(user_name=user_name)
    user = user_one.user_id
    stars = request.POST.get('ratingRadioOptions', None)
    print(user)
    print(stars)
    rating = Rating(user_id=user, recipe_id=id, selected_recipe_name=recipe_one.title, stars=stars)
    rated_stars = Rating.objects.filter(user_id=user).filter(recipe_id=id)
    print(rated_stars)
    if rated_stars != '<QuerySet []>':
        rating.save()
    else:
        pass
    return redirect('/signup_5/')


def signup_rate_4(request, id):
    recipe_one = Recipe.objects.get(recipe_id=id)
    user_name = request.session['user_name']
    user_one = UserInfo.objects.get(user_name=user_name)
    user = user_one.user_id
    stars = request.POST.get('ratingRadioOptions', None)
    print(user)
    print(stars)
    rating = Rating(user_id=user, recipe_id=id, selected_recipe_name=recipe_one.title, stars=stars)
    rated_stars = Rating.objects.filter(user_id=user).filter(recipe_id=id)
    print(rated_stars)
    if rated_stars != '<QuerySet []>':
        rating.save()
    else:
        pass
    return redirect('/')


def about_us(request):
    graph = visualize_cluster_3d()
    return render(request, 'about_us.html', {'graph': graph})


def pinned_recipe(request):
    today = datetime.today().strftime('%Y-%m-%d')
    yesterday_get = datetime.today() - timedelta(days=1)
    yesterday = yesterday_get.strftime('%Y-%m-%d')

    user = request.session['user']
    pinned_all = PinnedRecipe.objects.filter(user_id=user)
    # pinned_all = PinnedRecipe.objects.select_related('recipe')

    # Recipes_list = Recipe.objects.all()
    paginator = Paginator(pinned_all, 12)
    try:
        page = int(request.GET.get('page', '1'))
    except:
        page = 1
    try:
        Recipes = paginator.page(page)
    except(EmptyPage, InvalidPage):
        Recipes = paginator.page(paginator.num_pages)

    return render(request, 'pinned_recipe.html', {'list': Recipes})


def search_result(request):
    recipes_list = Recipe.objects.all()
    paginator = Paginator(recipes_list, 12)
    try:
        page = int(request.GET.get('page', '1'))
    except:
        page = 1
    try:
        recipes = paginator.page(page)
    except(EmptyPage, InvalidPage):
        recipes = paginator.page(paginator.num_pages)

    return render(request, 'search_result.html', {'Recipes': recipes})


def search_result_q(request):
    recipes = None
    query = None
    selected = None
    ingredient_list = None

    if 'q' in request.GET:
        query = request.GET.get('q')
        category_list = request.GET.get('category')
        ingredient_list1 = request.GET.get('ingredient1')
        ingredient_list2 = request.GET.get('ingredient2')

        # __icontains : 대소문자 구분없이 필드값에 해당 query가 있는지 확인 가능
        recipes = Recipe.objects.all().filter(Q(title__icontains=query) | Q(ingredients__icontains=query))\
            .filter(Q(category__icontains=category_list))\
            .exclude(Q(title__icontains=ingredient_list1) | Q(ingredients__icontains=ingredient_list1))\
            .exclude(Q(title__icontains=ingredient_list2) | Q(ingredients__icontains=ingredient_list2))

    paginator = Paginator(recipes, 12)
    try:
        page = int(request.GET.get('page', '1'))
    except:
        page = 1
    try:
        recipes = paginator.page(page)
    except(EmptyPage, InvalidPage):
        recipes = paginator.page(paginator.num_pages)

    return render(request, 'search_result_q.html', {'query': query, 'Recipes': recipes})


# %% 알고리즘 테스트 영역
def algorithm(request):
    if request.method == 'GET':
        return render(request, 'algorithm.html')


# %%
def show_CBF(request):
    user_id = request.POST['user_id']
    print(user_id)
    CBF(int(user_id))

    # 2초안에 검색결과가 나오게 하고 안되면 2초를 더 줌
    try:
        sleep(2)
        with open(BASE_DIR + '/output/CBF_Recommender/' + 'User_ID_' + str(user_id) + '_CBF_results.json', 'r', encoding='utf-8') as f:
            recommender_json = json.load(f)

    except:
        sleep(2)
        with open(BASE_DIR + '/output/CBF_Recommender/' + 'User_ID_' + str(user_id) + '_CBF_results.json', 'r', encoding='utf-8') as f:
            recommender_json = json.load(f)

    # json에서 각 열을 list 형식으로 담아옴
    recommended_recipe = list(recommender_json['recommended_recipe'].values())
    user_preferred_recipe = list(recommender_json['user_preferred_recipe'].values())
    ingredients_cosine_similarity = list(recommender_json['ingredients_cosine_similarity'].values())

    return render(request, 'algorithm_manage/Show_CBF.html',
                  {'recommended_recipe': recommended_recipe, 'user_preferred_recipe': user_preferred_recipe,
                   'ingredients_cosine_similarity': ingredients_cosine_similarity})


# %%
def show_CF(request):
    user_id = request.POST['user_id']
    print(user_id)
    CF(int(user_id))

    # 2초안에 검색결과가 나오게 하고 안되면 2초를 더 줌
    try:
        sleep(2)
        with open(BASE_DIR + '/output/CF_Recommender/' + 'User_ID_' + str(user_id) + '_CF_results.json', 'r', encoding='utf-8') as f:
            recommender_json = json.load(f)

    except:
        sleep(2)
        with open(BASE_DIR + '/output/CF_Recommender/' + 'User_ID_' + str(user_id) + '_CF_results.json', 'r', encoding='utf-8') as f:
            recommender_json = json.load(f)

    # dataframe에서 각 열을 list 형식으로 담아옴
    recommended_recipe = list(recommender_json['recommended_recipe'].values())
    user_preferred_recipe = list(recommender_json['user_preferred_recipe'].values())

    return render(request, 'algorithm_manage/Show_CF.html',
                  {'recommended_recipe': recommended_recipe, 'user_preferred_recipe': user_preferred_recipe})


def show_rating(request):
    # user_idf를 정수로
    user_id = request.POST['user_id']
    user_id = int(user_id)
    # Rating 정보를 DB에서 불러옴
    download_rating()
    rating = pd.read_json(BASE_DIR + '/output/user_dummy_data')

    # json에서 각 열을 list 형식으로 담아옴
    user_rating = rating[rating['user_id'] == user_id]
    selected_recipe_name = list(user_rating['selected_recipe_name'].tolist())
    stars = list(user_rating['stars'].tolist())

    return render(request, 'algorithm_manage/Show_Rating.html',
                  {'selected_recipe_name': selected_recipe_name, 'stars': stars})


# %% 모델 업데이트 및 더메 데이터 제작
# %% 클러스터링 업데이트
def update_cluster(request):
    make_clusters()
    return render(request, 'algorithm.html')


# %% CBF 모델 업데이트
def update_CBF(request):
    make_CBF_model()
    return render(request, 'algorithm.html')


# %% CF 모델 업데이트
def update_CF(request):
    make_CF_model()
    return render(request, 'algorithm.html')


# %% 더미 데이터 제작하기
def make_dummy(request):
    make_dummy_5stars()
    return render(request, 'algorithm.html')


# %%
def recommend_by_algorithm(request):
    user = request.session['user']
    USER_ID = user
    print('user: ', user)
    print('USER_ID: ', USER_ID)
    recommended_recipe_CBF = recommended_recipe_data_by_CBF(user_id=USER_ID)
    recommended_recipe_CF = recommended_recipe_data_by_CF(user_id=USER_ID)

    for i in range(len(recommended_recipe_CBF)):
        globals()['recipe_{}'.format(i + 1)] = dict(
            zip(list(recommended_recipe_CBF.columns), tuple(recommended_recipe_CBF.iloc[i])))

        # 카테고리명을 category 지역구분과 재료 구분으로 분리함
        globals()['recipe_{}'.format(i + 1)]['category_region'] = globals()['recipe_{}'.format(i + 1)]['category'].split('<')[0].strip()
        try:
            globals()['recipe_{}'.format(i + 1)]['category_integredients'] = globals()['recipe_{}'.format(i + 1)]['category'].split('<')[1].split(':')[1].replace('>', '').strip()
        except:
            globals()['recipe_{}'.format(i + 1)]['category_integredients'] = None

    recipe_lists = []
    for i in range(len(recommended_recipe_CBF)):
        recipe_lists.append(globals()['recipe_{}'.format(i + 1)])

    for i in range(len(recommended_recipe_CF)):
        globals()['recipe_{}'.format(i + 1)] = dict(zip(list(recommended_recipe_CF.columns), tuple(recommended_recipe_CF.iloc[i])))

        # 카테고리명을 category 지역구분과 재료 구분으로 분리함
        globals()['recipe_{}'.format(i + 1)]['category_region'] = globals()['recipe_{}'.format(i + 1)]['category'].split('<')[0].strip()
        try:
            globals()['recipe_{}'.format(i + 1)]['category_integredients'] = globals()['recipe_{}'.format(i + 1)]['category'].split('<')[1].split(':')[1].replace('>', '').strip()
        except:
            globals()['recipe_{}'.format(i + 1)]['category_integredients'] = None

    recipe_lists2 = []
    for i in range(len(recommended_recipe_CF)):
        recipe_lists2.append(globals()['recipe_{}'.format(i + 1)])

    # youtube
    today_video = today_yt()

    # twitter
    # today_twitter = today_tw()
    # 'today_tw': today_twitter

    # 통계 섹션
    counts = dict()

    # 통계 섹션 - 레시피 수
    recipeDF = pd.DataFrame(list(Recipe.objects.all().values()))
    counts['recipe_count'] = recipeDF['recipe_id'].count()

    # 통계 섹션 - 유저 수
    user_all = UserInfo.objects.all()
    counts['user_count'] = len(user_all)

    # 통계 섹션 - 찜 레시피 수
    pinned_recipe_all = PinnedRecipe.objects.all()
    counts['pinned_recipe_count'] = len(pinned_recipe_all)

    return render(request, 'main_login.html', {'recipe_lists': recipe_lists, 'recipe_lists2': recipe_lists2, 'counts': counts, 'today_yt': today_video})


# %%
def main_login_q(request):
    user = request.session['user']
    user_id=user
    Recipes = None
    query = None
    selected = None
    ingredient_list = None

    if 'q' in request.GET:
        query = request.GET.get('q')
        category_list = request.GET.get('category')
        ingredient_list1 = request.GET.get('ingredient1')
        ingredient_list2 = request.GET.get('ingredient2')
        # __icontains : 대소문자 구분없이 필드값에 해당 query가 있는지 확인 가능
        recipes = Recipe.objects.all().filter(Q(category__icontains=category_list)) \
            .exclude(Q(title__icontains=ingredient_list1) | Q(ingredients__icontains=ingredient_list1))\
            .exclude(Q(title__icontains=ingredient_list2) | Q(ingredients__icontains=ingredient_list2))

    users_filter = pd.DataFrame(list(recipes))

    #paginator = Paginator(Recipes, 12)
    try:
        page = int(request.GET.get('page', '1'))
    except:
        page = 1
    # try:
    #     Recipes = paginator.page(page)
    # except(EmptyPage, InvalidPage):
    #     Recipes = paginator.page(paginator.num_pages)

    # save Recipe_df to json file
    users_filter.to_json(BASE_DIR+'/output/users_filter.json')

    def recommend_by_filtered_algorithm(request, user_id):
        recommended_recipe_CBF = filtered_recipe_data_by_CBF(user_id)
        recommended_recipe_CF = filtered_recipe_data_by_CF(user_id)

        for i in range(len(recommended_recipe_CBF)):
            globals()['recipe_{}'.format(i+1)]=dict(zip(list(recommended_recipe_CBF.columns),tuple(recommended_recipe_CBF.iloc[i])))

            # 카테고리명을 category 지역구분과 재료 구분으로 분리함
            globals()['recipe_{}'.format(i + 1)]['category_region'] = globals()['recipe_{}'.format(i + 1)]['category'].split('<')[0].strip()
            try:
                globals()['recipe_{}'.format(i + 1)]['category_integredients'] = globals()['recipe_{}'.format(i + 1)]['category'].split('<')[1].split(':')[1].replace('>', '').strip()
            except:
                globals()['recipe_{}'.format(i + 1)]['category_integredients'] = None

        recipe_lists = []
        for i in range(len(recommended_recipe_CBF)):
            recipe_lists.append(globals()['recipe_{}'.format(i+1)])

        for i in range(len(recommended_recipe_CF)):
            globals()['recipe_{}'.format(i+1)]=dict(zip(list(recommended_recipe_CF.columns),tuple(recommended_recipe_CF.iloc[i])))

            # 카테고리명을 category 지역구분과 재료 구분으로 분리함
            globals()['recipe_{}'.format(i + 1)]['category_region'] = globals()['recipe_{}'.format(i + 1)]['category'].split('<')[0].strip()
            try:
                globals()['recipe_{}'.format(i + 1)]['category_integredients'] = globals()['recipe_{}'.format(i + 1)]['category'].split('<')[1].split(':')[1].replace('>', '').strip()
            except:
                globals()['recipe_{}'.format(i + 1)]['category_integredients'] = None

        recipe_lists2=[]
        for i in range(len(recommended_recipe_CF)):
            recipe_lists2.append(globals()['recipe_{}'.format(i+1)])

        return recipe_lists, recipe_lists2

    recipe_lists, recipe_lists2 = recommend_by_filtered_algorithm(request,user_id)
    return render(request, 'main_login_q.html', {'recipe_lists': recipe_lists, 'recipe_lists2': recipe_lists2})
