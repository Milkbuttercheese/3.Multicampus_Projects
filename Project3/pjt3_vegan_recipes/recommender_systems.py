from .BASE_DIR import BASE_DIR
import pandas as pd
import numpy as np
import re
from sqlalchemy import create_engine

import gensim
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from gensim.test.utils import get_tmpfile

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dot, Add, Flatten, Dense, Concatenate, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import SGD, Adam, Adamax

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.utils import shuffle
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
from plotly.offline import plot

plt.rcParams.update({'font.family': 'AppleGothic'})


# 텍스트에 포함된 특수문자 제거 함수
# 단 &는 재료의 최소단위를 구분짓는 경계로 사용할 것이기 때문에 제외
def remove_special_char(read_data):
    text = re.sub('[-=+,#/\?:^.@*\"※~%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》|\u0080-\uffef]', ' ', read_data)
    return text


# 숫자 제거 함수
def remove_num(read_data):
    text = re.sub(r'[0-9]+', '', read_data)
    return text


# 유니코드 제거 함수 (예:†¼½¾⅓⅔)
def remove_unicode(read_data):
    # encode() method
    strencode = read_data.encode("ascii", "replace")
    # decode() method
    strdecode = strencode.decode()
    return strdecode


# 총 특수문자/숫자/유니코드 제거하도록 통합시킨 함수
def preprocess_text(read_data):
    read_data = remove_unicode(read_data)
    read_data = remove_special_char(read_data)
    result = remove_num(read_data)
    return result


# %% 기본 설정값
# 클러스터링 분석에 사용될 재료를 빈도수로 몇개를 택할 것인가?
num_selected_feature = 100
# 클러스터의 수
num_cluster = 6
# 더미 유저 수
num_dummy_user = 20000
# 더미 레시피 수
num_dummy_recipe = 200
# 추천 레시피 수
top_n = 20


# %%
# 1.클러스터링

# %% 1-1. 데이터 셋 불러오기
# 파이썬에서 MySql 연결을 위한 함수
def download_recipes():
    table_nm = 'recipe'
    user_nm = 'root'
    user_pw = 't0101'
    host_nm = '35.79.107.247'
    host_address = '3306'
    db_nm = 'team01'
    # 데이터베이스 연결
    db_connection_path = f'mysql+mysqldb://{user_nm}:{user_pw}@{host_nm}:{host_address}/{db_nm}'
    db_connection = create_engine(db_connection_path, encoding='utf-8')
    conn = db_connection.connect()
    # 데이터 로딩
    df = pd.read_sql_table(table_nm, con=conn)
    df['ingredients'] = df['ingredients'].apply(lambda x: x.split(','))
    return df


def upload_dataset(df, table_nm):
    user_nm = 'root'
    user_pw = 't0101'
    host_nm = '35.79.107.247'
    host_address = '3306'
    db_nm = 'team01'
    # 데이터베이스 연결
    db_connection_path = f'mysql+mysqldb://{user_nm}:{user_pw}@{host_nm}:{host_address}/{db_nm}'
    db_connection = create_engine(db_connection_path, encoding='utf-8')
    conn = db_connection.connect()
    # 데이터 적재
    df.to_sql(name=table_nm, con=conn, if_exists='replace')
    return df


# %% 1-2.레시피 총 데이터셋에서 title,ingredients 열만 추출, ingredients를 클러스터링을 위해 전처리 하기
def C2_get_preprocessed_recipe(df):
    # 레시피와 재료만 추출
    recipe_N_ingredients = df[['title', 'ingredients']]

    # 재료 데이터의 복잡한 데이터구조 [레시피:[재료]] [재료1,재료2,...,재료_n] 재료_j는 str거나 dict인 구조.
    # 재료가 dict인 경우 안의 원재료명만 추출하는 방식을 사용

    ingredients_lst = [' ' for i in range(len(recipe_N_ingredients))]
    title_lst = []

    for i in range(len(recipe_N_ingredients)):
        title_lst.append(recipe_N_ingredients.iloc[i]['title'])
        try:
            for j in range(len(recipe_N_ingredients.iloc[i]['ingredients'])):
                if type(recipe_N_ingredients.iloc[i]['ingredients'][j]) == str:
                    ingredients_lst[i] = ingredients_lst[i] + preprocess_text(
                        str(recipe_N_ingredients.iloc[i]['ingredients'][j])) + ' '
                    ingredients_lst[i] = re.sub(',', ' ', ingredients_lst[i])

                elif type(recipe_N_ingredients.iloc[i]['ingredients'][j]) == dict:
                    ingredients_lst[i] = ingredients_lst[i] + \
                                         preprocess_text(str([x + '  ' for x in list(
                                             recipe_N_ingredients.iloc[i]['ingredients'][j].values())[0]])) + ' '
                    ingredients_lst[i] = re.sub(',', ' ', ingredients_lst[i])
        except:
            ingredients_lst[i] = 'Not Found'

    # 전처리된 결과를 데이터프레임에 담기
    recipe_N_ingredients_2 = pd.DataFrame([title_lst, ingredients_lst], index=['title', 'ingredients'])
    recipe_N_ingredients_2 = recipe_N_ingredients_2.T
    recipe_N_ingredients_2['ingredients'] = recipe_N_ingredients_2['ingredients'].apply(lambda x: x.lower())

    # 요리 측량 단위 레퍼런스: https://en.wikibooks.org/wiki/Cookbook:Units_of_measurement
    # 요리 측정 단위 단어들을 불용어로 지정하기 위해 다음과 같은 단어 리스트들을 지정해둔다
    ingredient_stopwords = ['fresh', 'optional', 'sliced', 'cubes', 'hot', 'frozen', 'juiced', 'syrup', 'taste',
                            'unsweetened', 'soft', 'removed', 'plant', 'based', 'choice', 'tspground', 'turmeric',
                            'pinchground', 'black', 'canned', 'granulated', 'vegan', 'pure', 'extract', 'brown',
                            'boilng', 'powder', 'syrupagave', 'crushed', 'whole', 'cloves', 'dairy', 'free', 'dark',
                            'drained', 'ground', 'medium', 'vegatable', 'bouillon', 'cooked', 'small', 'yellow', 'bell',
                            'sauce', 'nutritional', 'chopped', 'red', 'cut', 'thinly', 'dry', 'white', 'baking',
                            'minced', 'dried', 'peeled', 'purpose', 'roasted', 'oregano', 'rolled', 'diced', 'raw',
                            'extra', 'large', 'water', 'leaves', 'green', 'plus', 'juice', 'light', 'divided', 'melted',
                            'plain', 'rinsed', 'fat', 'seeds', 'toasted', 'clove', 'flakes', 'shredded', 'finely',
                            'grated', 'roughly', 'freshly', 'sweet', 'sea', 'packed', 'ripe', 'like', 'virgin',
                            'smoked', 'organic', 'bought', 'use', 'needed', 'serve', 'pinch', 'recipe', 'sub', 'gf',
                            'adjust', 'ounces', 'tablespoons', 'handful', 'used', 'teaspoons', 'chips', 'slices',
                            'pieces', 'less', 'soaked', 'half', 'pitted', 'low', 'thin', 'store',
                            'baby', 'see', 'kosher', 'non', 'fine', 'not', 'found', 'tablespoon', 'ice', 'cooking',
                            'full', 'firm', 'gluten', 'paste', 'garnish', 'bunch', 'yields', 'written', 'halved',
                            'stock', 'spice', 'mix', 'cayenne', 'spray', 'spring', 'heaped', 'vegetable', 'powdered',
                            'topping', 'mixed', 'caster']

    volume = ['ml', 'mL', 'cc', 'l', 'L', 'liter', 'dl', 'dL', 'teaspoon', 't', 'tsp', 'tablespoon', 'T', 'tbl', 'tbs',
              'tsp', 'Tbsp', 'tbsp', 'fl oz', 'gill', 'cup', 'cups', 'c', 'pint', 'p', 'pt', 'fl pt', 'quart',
              'q', 'qt', 'fl qt', 'gallon', 'g', 'gal', 'large', 'small', 'medium', 'half']

    weight = ['mg', 'g', 'gram', 'kg', 'pound', 'lb', 'ounce', 'oz', 'lbs', 'pounds']
    length = ['mm', 'cm', 'm', 'inch', 'in', '\"', 'yard', 'inches', 'length']
    temperature = ['°C', '°F', 'C', 'F']
    time_ = ['year', 'years', 'month', 'weeks', 'week', 'days', 'day', 'hours', 'hour', 'mintus', 'seconds', 'second']
    etc = [' ']

    adjective = ['diced', 'divided', 'Raw', 'sized', 'yellow', 'white', 'White', 'black', 'heavy', 'mature', 'sub',
                 'trimmed', 'top', 'Peeled', 'delicious', 'one']

    preposition = ['depending']
    verb = ['cut', 'see', 'note', 'use']
    noun = ['Cups', 'temperature', 'Temperature']

    # medium, cut, yellow, white, Cups, heavy ,length , mature, black, half, room, see, note, use

    measurements = volume + weight + length + temperature + time_ + etc + adjective + preposition + verb + noun + ingredient_stopwords
    set_measurements = set(measurements)

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    stop_words = stop_words | set_measurements

    # 리스트 내 원소들을 합치기용 함수 - apply를 통해 row 단위에 적용
    def lst_to_str(lst):
        result = ''
        for item in lst:
            result = result + ' ' + item
        return result

    # 단어들을 띄어쓰기 단위로 쪼개기 => 하나의 레시피에 하나의 재료 리스트가 대응되게 됨
    splited_sr = pd.Series(recipe_N_ingredients_2['ingredients']).apply(lambda x: x.split())
    # 불용어를 이용한 필터링
    filtered_sr = splited_sr.apply(lambda x: [item for item in x if item not in stop_words])
    # 행에 중복된 단어 삭제
    unique_df = filtered_sr.apply(lambda x: list(set(x)))
    # stemming
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    lemmatizer = WordNetLemmatizer()
    lemmatized_df = unique_df.apply(lambda x: [lemmatizer.lemmatize(word, 'n') for word in x])

    # 다시 리스트를 문자열로 합치기
    preprocessed_sr = lemmatized_df.apply(lst_to_str)
    # 이제 원래 목표인 '&를 경계로 재료명을 구분짓기'를 시행
    tokened_sr = preprocessed_sr.apply(lambda x: x.split('&'))

    tokened_df = pd.DataFrame([title_lst, list(preprocessed_sr.values)], index=['title', 'ingredients'])
    tokened_df = tokened_df.T

    return tokened_df, recipe_N_ingredients_2


# %% 1-3.빈도순으로 주요 단어 100개를 선정하고, 주요단어에 대한 레시피들의 TF-IDF를 계산하기
# 추후 GMM 같은 다른 알고리즘으로 바꿀 수 있는지 다시 한번 확인해보자
def C3_TF_IDF(tokened_df):
    token_lst = tokened_df['ingredients'].tolist()
    vocab = list(set(w for doc in token_lst for w in doc.split()))

    ingredient_stopwords = ['fresh', 'optional', 'sliced', 'cubes', 'hot', 'frozen', 'juiced', 'syrup', 'taste',
                            'unsweetened', 'soft', 'removed', 'plant', 'based', 'choice', 'tspground', 'turmeric',
                            'pinchground', 'black', 'canned', 'granulated', 'vegan', 'pure', 'extract', 'brown',
                            'boilng', 'powder', 'syrupagave', 'crushed', 'whole', 'cloves', 'dairy', 'free', 'dark',
                            'drained', 'ground', 'medium', 'vegatable', 'bouillon', 'cooked', 'small', 'yellow', 'bell',
                            'sauce', 'nutritional', 'chopped', 'red', 'cut', 'thinly', 'dry', 'white', 'baking',
                            'minced', 'dried', 'peeled', 'purpose', 'roasted', 'oregano', 'rolled', 'diced', 'raw',
                            'extra', 'large', 'water', 'leaves', 'green', 'plus', 'juice', 'light', 'divided', 'melted',
                            'plain', 'rinsed', 'fat', 'seeds', 'toasted', 'clove', 'flakes', 'shredded', 'finely',
                            'grated', 'roughly', 'freshly', 'sweet', 'sea', 'packed', 'ripe', 'like', 'virgin',
                            'smoked', 'organic', 'bought', 'use', 'needed', 'serve', 'pinch', 'recipe', 'sub', 'gf',
                            'adjust', 'ounces', 'tablespoons', 'handful', 'used', 'teaspoons', 'chips', 'slices',
                            'pieces', 'less', 'soaked', 'half', 'pitted', 'low', 'thin', 'store',
                            'baby', 'see', 'kosher', 'non', 'fine', 'not', 'found', 'tablespoon', 'ice', 'cooking',
                            'full', 'firm', 'gluten', 'paste', 'garnish', 'bunch', 'yields', 'written', 'halved',
                            'stock', 'spice', 'mix', 'cayenne', 'spray', 'spring', 'heaped', 'vegetable', 'powdered',
                            'topping', 'mixed', 'caster']

    tfidfv = TfidfVectorizer(max_features=num_selected_feature, stop_words=ingredient_stopwords).fit(token_lst)

    # selected_feature의 TF_IDF matrix 계산하기
    TF_IDF_matrix = tfidfv.transform(token_lst).toarray()

    kmeans = KMeans(n_clusters=num_cluster, init='k-means++', max_iter=300, random_state=0)
    kmeans.fit(TF_IDF_matrix)
    TF_IDF_matrix = TF_IDF_matrix.astype(float)
    TF_IDF_matrix = pd.DataFrame(TF_IDF_matrix)
    TF_IDF_matrix['cluster'] = kmeans.labels_

    # 나중에 빈도순 중요 단어가 무엇인지 알려줄 때 사용
    vocabs = tfidfv.vocabulary_
    return TF_IDF_matrix, tfidfv, vocabs


# %% 1-R1.여러개의 클러스터링 개수를 list로 입력받아 실루엣 계수를 시각화하는 함수
# X_features= TF_IDF_matrix.iloc[:,0:selected_feature] 필요
def visualize_silhouette(cluster_lists):
    df = download_recipes()
    tokened_df, recipe_N_ingredients_2 = C2_get_preprocessed_recipe(df)
    TF_IDF_matrix, tfidfv, vocabs = C3_TF_IDF(tokened_df)
    X_features = TF_IDF_matrix.iloc[:, 0:num_selected_feature]

    # 입력값으로 클러스터링 갯수들을 리스트로 받아서, 각 갯수별로 클러스터링을 적용하고 실루엣 개수를 구함
    n_cols = len(cluster_lists)

    # plt.subplots()으로 리스트에 기재된 클러스터링 수만큼의 sub figures를 가지는 axs 생성
    plt.figure(dpi=300)
    fig, axs = plt.subplots(figsize=(4 * n_cols, 4), nrows=1, ncols=n_cols)

    # 리스트에 기재된 클러스터링 갯수들을 차례로 iteration 수행하면서 실루엣 개수 시각화
    for ind, num_cluster in enumerate(cluster_lists):

        # KMeans 클러스터링 수행하고, 실루엣 스코어와 개별 데이터의 실루엣 값 계산.
        clusterer = KMeans(n_clusters=num_cluster, max_iter=500, random_state=0)
        cluster_labels = clusterer.fit_predict(X_features)

        sil_avg = silhouette_score(X_features, cluster_labels)
        sil_values = silhouette_samples(X_features, cluster_labels)

        y_lower = 10
        axs[ind].set_title('Number of Cluster : ' + str(num_cluster) + '\nSilhouette Score :' + str(round(sil_avg, 3)))
        axs[ind].set_xlabel("The silhouette coefficient values")
        axs[ind].set_ylabel("Cluster label")
        axs[ind].set_xlim([-0.1, 1])
        axs[ind].set_ylim([0, len(X_features) + (num_cluster + 1) * 10])
        axs[ind].set_yticks([])  # Clear the yaxis labels / ticks
        axs[ind].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])

        # 클러스터링 갯수별로 fill_betweenx( )형태의 막대 그래프 표현.
        for i in range(num_cluster):
            ith_cluster_sil_values = sil_values[cluster_labels == i]
            ith_cluster_sil_values.sort()

            size_cluster_i = ith_cluster_sil_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / num_cluster)
            axs[ind].fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_sil_values, facecolor=color, edgecolor=color, alpha=0.7)
            axs[ind].text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
            y_lower = y_upper + 10

        axs[ind].axvline(x=sil_avg, color="red", linestyle="--")

    plt.savefig(BASE_DIR + '/output/Clustering/silhouette.jpg')


# %% 1-R2. 클러스터링하고, 그 결과를 json파일로 저장하기
# 레시피 df, 클러스터 df, 빈도수 주요단어를 반환한다
def make_clusters():
    df = download_recipes()
    tokened_df, recipe_N_ingredients_2 = C2_get_preprocessed_recipe(df)
    TF_IDF_matrix, tfidfv, vocabs = C3_TF_IDF(tokened_df)

    # 클러스터로으로 나누어졌을 떄의 각각 클러스터별 특징 살피기
    Cluster_df = TF_IDF_matrix.iloc[:, 0:num_selected_feature + 1].groupby('cluster').mean()

    # 각각 클러스터별 상위 ingredients_len개 재료 선택
    ingredieints_len = 10
    cluster1_feature = list(Cluster_df.iloc[0].sort_values(ascending=False).index)[0:ingredieints_len]
    cluster2_feature = list(Cluster_df.iloc[1].sort_values(ascending=False).index)[0:ingredieints_len]
    cluster3_feature = list(Cluster_df.iloc[2].sort_values(ascending=False).index)[0:ingredieints_len]
    cluster4_feature = list(Cluster_df.iloc[3].sort_values(ascending=False).index)[0:ingredieints_len]
    cluster5_feature = list(Cluster_df.iloc[4].sort_values(ascending=False).index)[0:ingredieints_len]
    cluster6_feature = list(Cluster_df.iloc[5].sort_values(ascending=False).index)[0:ingredieints_len]
    # cluster7_feature= list(Cluster_df.iloc[6].sort_values(ascending=False).index)[0:ingredieints_len]
    # cluster8_feature= list(Cluster_df.iloc[7].sort_values(ascending=False).index)[0:ingredieints_len]
    # cluster9_feature= list(Cluster_df.iloc[8].sort_values(ascending=False).index)[0:ingredieints_len]
    # cluster10_feature= list(Cluster_df.iloc[9].sort_values(ascending=False).index)[0:ingredieints_len]

    # 정수 인코딩된 재료명을 다시 원래의 자연어 재료명으로 변환
    ingredients_int = list(tfidfv.vocabulary_.values())
    ingredients_name = list(tfidfv.vocabulary_.keys())
    ingredient_df = pd.DataFrame([ingredients_int, ingredients_name], index=['인코딩된 정숫값', '재료명']).T
    ingredient_df = ingredient_df.sort_values(by='인코딩된 정숫값').set_index('인코딩된 정숫값')

    # 각 재료명을 추출 후 데이터프레임에 담기
    cluster1_df = ingredient_df.iloc[cluster1_feature].values.flatten()
    cluster2_df = ingredient_df.iloc[cluster2_feature].values.flatten()
    cluster3_df = ingredient_df.iloc[cluster3_feature].values.flatten()
    cluster4_df = ingredient_df.iloc[cluster4_feature].values.flatten()
    cluster5_df = ingredient_df.iloc[cluster5_feature].values.flatten()
    cluster6_df = ingredient_df.iloc[cluster6_feature].values.flatten()
    # cluster7_df= ingredient_df.iloc[cluster7_feature].values.flatten()
    # cluster8_df= ingredient_df.iloc[cluster8_feature].values.flatten()
    # cluster9_df= ingredient_df.iloc[cluster9_feature].values.flatten()
    # cluster10_df= ingredient_df.iloc[cluster9_feature].values.flatten()

    clusters_df = pd.DataFrame([cluster1_df, cluster2_df, cluster3_df, cluster4_df, cluster5_df, cluster6_df],
                               index=['클러스터1', '클러스터2', '클러스터3', '클러스터4', '클러스터5', '클러스터6'])  #

    # 각 클러스터에 대응되는 레시피 붙이기
    # 군집값이 0,1,2,3,4,5,6,7 이니 경우 마다 별도의 인덱스로 추출
    cluster1_index = TF_IDF_matrix[TF_IDF_matrix['cluster'] == 0].index.tolist()
    cluster2_index = TF_IDF_matrix[TF_IDF_matrix['cluster'] == 1].index.tolist()
    cluster3_index = TF_IDF_matrix[TF_IDF_matrix['cluster'] == 2].index.tolist()
    cluster4_index = TF_IDF_matrix[TF_IDF_matrix['cluster'] == 3].index.tolist()
    cluster5_index = TF_IDF_matrix[TF_IDF_matrix['cluster'] == 4].index.tolist()
    cluster6_index = TF_IDF_matrix[TF_IDF_matrix['cluster'] == 5].index.tolist()
    # cluster7_index= TF_IDF_matrix[TF_IDF_matrix['cluster']==6].index.tolist()
    # cluster8_index= TF_IDF_matrix[TF_IDF_matrix['cluster']==7].index.tolist()
    # cluster9_index= TF_IDF_matrix[TF_IDF_matrix['cluster']==8].index.tolist()
    # cluster10_index= TF_IDF_matrix[TF_IDF_matrix['cluster']==9].index.tolist()

    ingredient_names = recipe_N_ingredients_2['title']
    cluster1_title = ingredient_names.iloc[cluster1_index].values.tolist()
    cluster2_title = ingredient_names.iloc[cluster2_index].values.tolist()
    cluster3_title = ingredient_names.iloc[cluster3_index].values.tolist()
    cluster4_title = ingredient_names.iloc[cluster4_index].values.tolist()
    cluster5_title = ingredient_names.iloc[cluster5_index].values.tolist()
    cluster6_title = ingredient_names.iloc[cluster6_index].values.tolist()
    # cluster7_title=ingredient_names.iloc[cluster7_index].values.tolist()
    # cluster8_title=ingredient_names.iloc[cluster8_index].values.tolist()
    # cluster9_title=ingredient_names.iloc[cluster9_index].values.tolist()
    # cluster10_title=ingredient_names.iloc[cluster10_index].values.tolist()

    clusters_df['recipe'] = [cluster1_title, cluster2_title, cluster3_title, cluster4_title, cluster5_title,
                             cluster6_title]

    ## 클러스터 네이밍
    #  특정 재료가 3개 이상 있으면 해당 클러스터에 네이밍을 한다
    cluster_indian = None
    cluster_asian = None
    cluster_western = None
    cluster_dessert = None
    indian_index = []
    asian_index = []
    western_index = []
    dessert_index = []

    for i in range(1, num_cluster + 1):
        if clusters_df.loc[f'클러스터{i}'].str.contains("lime|coriander|cilantro|cumin|avocado|onion").sum() >= 3:
            if type(cluster_western) is None:
                cluster_indian = clusters_df.loc[f'클러스터{i}']
                indian_index = indian_index + TF_IDF_matrix[TF_IDF_matrix['cluster'] == (i - 1)].index.tolist()

            else:
                cluster_indian = pd.concat([cluster_indian, clusters_df.loc[f'클러스터{i}']], axis=1)
                indian_index = indian_index + TF_IDF_matrix[TF_IDF_matrix['cluster'] == (i - 1)].index.tolist()

        elif clusters_df.loc[f'클러스터{i}'].str.contains("sesame|rice|soy|tofu").sum() >= 3:
            if type(cluster_western) is None:
                cluster_asian = clusters_df.loc[f'클러스터{i}']
                asian_index = asian_index + TF_IDF_matrix[TF_IDF_matrix['cluster'] == (i - 1)].index.tolist()

            else:
                cluster_asian = pd.concat([cluster_asian, clusters_df.loc[f'클러스터{i}']], axis=1)
                asian_index = asian_index + TF_IDF_matrix[TF_IDF_matrix['cluster'] == (i - 1)].index.tolist()

        elif clusters_df.loc[f'클러스터{i}'].str.contains(
                "sugar|milk|coconut|vanilla|butter|almond|cinnamon|yogurt").sum() >= 3:
            if type(cluster_western) is None:
                cluster_dessert = clusters_df.loc[f'클러스터{i}']
                dessert_index = dessert_index + TF_IDF_matrix[TF_IDF_matrix['cluster'] == (i - 1)].index.tolist()

            else:
                cluster_dessert = pd.concat([cluster_dessert, clusters_df.loc[f'클러스터{i}']], axis=1)
                dessert_index = dessert_index + TF_IDF_matrix[TF_IDF_matrix['cluster'] == (i - 1)].index.tolist()

        else:
            if type(cluster_dessert) is None:
                cluster_western = clusters_df.loc[f'클러스터{i}']
                western_index = western_index + TF_IDF_matrix[TF_IDF_matrix['cluster'] == (i - 1)].index.tolist()

            else:
                cluster_western = pd.concat([cluster_western, clusters_df.loc[f'클러스터{i}']], axis=1)
                western_index = western_index + TF_IDF_matrix[TF_IDF_matrix['cluster'] == (i - 1)].index.tolist()

    cluster_indian = cluster_indian.T
    cluster_asian = cluster_asian.T
    cluster_western = cluster_western.T
    cluster_dessert = cluster_dessert.T

    cluster_indian['클러스터'] = '1.인도+남아시아+남미 <주재료: 큐민/고수/라임/아보카도/양파>'
    cluster_asian['클러스터'] = '2.동아시아 <주재료: 쌀/간장/참깨/두부>'
    cluster_dessert['클러스터'] = '3.디저트+제과제빵 <주재료: 설탕/우유/코코넛/바닐라/버터/아몬드>'
    cluster_western['클러스터'] = '4.서양+기타'

    after_cluser = pd.concat([cluster_indian, cluster_asian, cluster_western, cluster_dessert])

    # 저장하기
    # 각 카테고리 명 붙이기
    # [식 for 변수1 in 리스트1 if 조건식1     for 변수2 in 리스트2 if 조건식2     ...     for 변수n in 리스트n if 조건식n]
    # n if n>0 else 0 for n in array

    cluster_lst = []

    for i in range(len(tokened_df)):
        if i in indian_index:
            cluster_lst.append('1.India+South America+South Asia <Main ingredients: '
                               'cumin/coriander/cilantro/lime/avocado/onion>')
        elif i in asian_index:
            cluster_lst.append('2.East Asia <Main ingredients: rice/soy/sesame/tofu>')
        elif i in dessert_index:
            cluster_lst.append('3.Dessert+Confectionery <Main ingredients: sugar/milk/coconut/vanilla/butter/almond>')
        else:
            cluster_lst.append('4.West+Etc')

    df.rename(columns={'site': 'link'}, inplace=True)
    df['category'] = pd.Series(cluster_lst)

    df['카테고리'] = pd.Series(cluster_lst)
    TF_IDF_matrix['cluster'] = pd.Series(cluster_lst)

    # 결과들 저장
    df.to_json(BASE_DIR + '/output/Clustering/preprocessed_recipes.json', orient='table', index=False)
    after_cluser.to_json(BASE_DIR + '/output/Clustering/clusters.json')
    vocabs = pd.DataFrame(vocabs, index=list(range(len(vocabs))))
    vocabs.to_json(BASE_DIR + '/output/Clustering/main_keywords.json')
    TF_IDF_matrix.to_json(BASE_DIR + '/output/Clustering/TF_IDF_matrix.json')

    print('클러스터 업데이트가 완료되었습니다')


# %%
def visualize_cluster():
    # 기본 설정
    pca = PCA(n_components=2)

    TF_IDF_matrix = pd.read_json(BASE_DIR + '/output/Clustering/TF_IDF_matrix.json')
    pca_transformed = pca.fit_transform(TF_IDF_matrix.iloc[:, 0:num_selected_feature])

    TF_IDF_matrix['pca_x'] = pca_transformed[:, 0]
    TF_IDF_matrix['pca_y'] = pca_transformed[:, 1]

    fig = px.scatter(TF_IDF_matrix, x='pca_x', y='pca_y', color='cluster', width=800, height=600)
    fig.layout.legend.x = 0.01
    fig.layout.legend.y = -0.5

    fig.write_image(BASE_DIR + '/output/Clustering/cluster.jpg', scale=2)
    fig.show()


# %%
def visualize_cluster_3d():
    pca = PCA(n_components=3)
    TF_IDF_matrix = pd.read_json(BASE_DIR + '/output/Clustering/TF_IDF_matrix.json')
    pca_transformed = pca.fit_transform(TF_IDF_matrix.iloc[:, 0:num_selected_feature])

    TF_IDF_matrix['pca_x'] = pca_transformed[:, 0]
    TF_IDF_matrix['pca_y'] = pca_transformed[:, 1]
    TF_IDF_matrix['pca_z'] = pca_transformed[:, 2]

    fig = px.scatter_3d(TF_IDF_matrix, x='pca_x', y='pca_y', z='pca_z', color='cluster', template='plotly_white')

    fig.update_layout(legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                      scene=dict(xaxis_title="", yaxis_title="", zaxis_title=""))

    newnames = {
        '1.India+Spain+South America+South Asia <Main ingredients: cumin/coriander/cilantro/lime/avocado/onion>': 'Latin + South Asia',
        '2.East Asia <Main ingredients: rice/soy/sesame/tofu>': 'East Asia',
        '3.Dessert+Confectionery <Main ingredients: sugar/milk/coconut/vanilla/butter/almond>': 'Dessert',
        '4.West+Etc': 'West +@', }

    fig.for_each_trace(lambda t: t.update(name=newnames[t.name], legendgroup=newnames[t.name],
                                          hovertemplate=t.hovertemplate.replace(t.name, newnames[t.name])))
    plot_div = plot(fig, output_type='div')
    return plot_div


# %% 2 더미데이터 제작
# %% 2-1. 더미데이터 제작
def make_dummy_5stars():
    df = download_recipes()

    # 무작위 200개 레시피 추출하여 데이터프레임의 컬럼명으로 쓰기
    np.random.seed(1)
    random_sampled_recipe = np.random.choice(range(1, len(df)), num_dummy_recipe, replace=False)
    recipe_names = list(df.iloc[random_sampled_recipe]['title'].values)

    # 더미 유저수 결정하기
    dummy_df = pd.DataFrame(index=range(0, num_dummy_user), columns=recipe_names)
    # 리뷰 안 단 비율과 단 비율을 0.95: 0.05로 설정

    r = 0.05
    random_numbers = [
        np.random.choice(np.arange(6), len(recipe_names), p=[1 - r, 0.05 * r, 0.15 * r, 0.2 * r, 0.35 * r, 0.25 * r])
        for i in range(
            num_dummy_user)]

    # 무작위로 리뷰한 결과
    random_reviews = pd.DataFrame(random_numbers, columns=recipe_names)

    # 더미 고객 한명이 리뷰를 남긴 갯수
    # pd.DataFrame(random_numbers).sum(axis=1)

    random_reviews.to_csv(BASE_DIR + '/output/dummy_data.csv')

    print('더미 데이터 제작이 완료되었습니다')
    return random_reviews


# %% 2-2. 로컬에서 더미 데이터(2-1 결과물)를 가공하여 저장하기
def user_for_db():
    ratings = None
    try:
        # 레시피 평가 데이터(rating_matrix) 불러오기
        # 행: 사용자 ID
        # 열: 레시피 ID
        rating_dummy = pd.read_csv(BASE_DIR + '/output/dummy_data.csv', index_col=False)
        rating_dummy.rename(columns={'Unnamed: 0': 'user_id'}, inplace=True)
        rating_dummy.set_index('user_id', inplace=True)

        # 딥러닝 학습을 위한 레시피명 변경
        rating_dummy_int = rating_dummy.copy().T
        rating_dummy_int.reset_index(drop=True, inplace=True)
        rating_dummy_int = rating_dummy_int.T

        # rating_matrix의 형태 변환
        rating_dummy_int = rating_dummy_int.replace(0, np.NaN)
        ratings = rating_dummy_int.stack()

        # ratings 데이터프레임 생성
        ratings = pd.DataFrame(ratings)
        ratings.reset_index(inplace=True)
        ratings.rename(columns={'level_0': 'user_id', 'level_1': 'selected_recipe_id', 0: 'stars'}, inplace=True)

        # 200개로 추려진 요리 목록을 딕셔너리 형태로 담기
        dummy = pd.read_csv(BASE_DIR + '/output/dummy_data.csv')
        selected_recipes_names = list(dummy.columns)[1:]
        recipe_ranges = list(range(num_dummy_recipe))
        selected_recipes_dict = dict(zip(recipe_ranges, selected_recipes_names))

        recipe_ids = [selected_recipes_dict[i] for i in ratings['selected_recipe_id']]
        ratings['selected_recipe_name'] = recipe_ids
        ratings.drop('selected_recipe_id', axis=1, inplace=True)
        ratings.to_json(BASE_DIR + '/output/user_dummy_data')
        print('로컬에서 유저 데이터 가공이 완료되었습니다')

    except:
        print('사용자 데이터가 존재하지 않습니다')


# %% 2-3. DB에서 유저 데이터 불러오기
def download_rating(table_nm='rating'):
    user_nm = 'root'
    user_pw = 't0101'
    host_nm = '35.79.107.247'
    host_address = '3306'
    db_nm = 'team01'
    # 데이터베이스 연결
    db_connection_path = f'mysql+mysqldb://{user_nm}:{user_pw}@{host_nm}:{host_address}/{db_nm}'
    db_connection = create_engine(db_connection_path, encoding='utf-8')
    conn = db_connection.connect()
    # 데이터 로딩
    df = pd.read_sql_table(table_nm, con=conn)
    df.to_json(BASE_DIR + '/output/user_dummy_data')
    print('DB로부터 유저 데이터를 다운로드 완료하였습니다')


# %% 3
# %% 콘텐츠 기반 필터링
def CBF(User_ID, model_loc=BASE_DIR + '/output/CBF_Recommender/CBF_Model'):
    CBF_df = None

    ratings = pd.read_json(BASE_DIR + '/output/user_dummy_data')
    user_rating_lst = ratings[ratings['user_id'] == User_ID]
    user_rating_lst = user_rating_lst[user_rating_lst['stars'] >= 4]
    user_rating_lst = user_rating_lst['selected_recipe_name']
    user_rating_lst = user_rating_lst.tolist()



    # 모델 불러오기
    model = doc2vec.Doc2Vec.load(model_loc)
    # 임베딩 벡터 평균치로써 유저가 가장 좋아할만한 레시피 20개를 추천한다
    recommend_result = model.dv.most_similar(user_rating_lst, topn=top_n)



    # 이때 데이터는 (레시피명,유사도) 튜플 형태로 반환된다
    # 추천된 레시피와 유사도 점수를 분리해서 담기
    recipe_name = [recommend_result[i][0] for i in range(len(recommend_result))]
    similarity_score = [recommend_result[i][1] for i in range(len(recommend_result))]
    CBF_df = pd.DataFrame([recipe_name, similarity_score, user_rating_lst],
                          index=['recommended_recipe', 'ingredients_cosine_similarity', 'user_preferred_recipe']).T

    print('CBF', 4)

    CBF_df.to_json(BASE_DIR + '/output/CBF_Recommender/' + 'User_ID_' + str(User_ID) + '_CBF_results.json')


# %% 3-R2. CBF 추천 알고리즘 모델 파일 만들기
# 절대 경로 /각자 컴퓨터에 맞게 수정 부탁드립니다
def make_CBF_model():

    df = download_recipes()
    tokened_df, recipe_N_ingredients_2 = C2_get_preprocessed_recipe(df)
    # 레시피-재료 document를 doc2vec 하여 레시피간 재료의 유사도를 고려하는 모델 생성하기
    # doc2Vec을 적용하기 위해 문자열로 구성된 ingredient 성분들을 띄워쓰기를 기준으로 list로 쪼갠다
    splited_lst = pd.Series(tokened_df['ingredients']).apply(lambda x: x.split()).tolist()
    # doc2Vec을 적용시키기 위한 데이터구조로 만들고, 적용시킨다
    # 이때 모델학습에 사용되는 데이터는 레시피 데이터셋이다
    taggedDocs = [TaggedDocument(words=splited_lst[i], tags=tokened_df['title'][{i}]) for i in range(len(splited_lst))]
    # Doc2Ve
    # 레시피 데이터셋을 활용하여 학습하였다
    # 각 레시피는 구성되는 재료들을 활용하여 유사도를 측정한다
    model = gensim.models.doc2vec.Doc2Vec(taggedDocs, dm=1, vector_size=50, epochs=10, hs=0, seed=0)
    model.train(taggedDocs, total_examples=model.corpus_count, epochs=model.epochs)
    # 모델 저장하기
    fname = get_tmpfile(BASE_DIR + '/output/CBF_Recommender/CBF_Model')
    model.save(fname)
    print('CBF 모델이 업데이트 완료되었습니다')


# %%
def metric_CBF(model_loc=BASE_DIR + '/output/CBF_Recommender/CBF_Model'):
    total_similarity_score = []
    ratings = pd.read_json(BASE_DIR + '/output/user_dummy_data')
    for User_ID in range(1, 20001):
        print(User_ID)
        user_rating_lst = ratings[ratings['user_id'] == User_ID]
        user_rating_lst = user_rating_lst[user_rating_lst['stars'] >= 4]
        user_rating_lst = user_rating_lst['selected_recipe_name']
        user_rating_lst = user_rating_lst.tolist()

        # 모델 불러오기
        try:
            model = doc2vec.Doc2Vec.load(model_loc)
            # 임베딩 벡터 평균치로써 유저가 가장 좋아할만한 레시피 10개를 추천한다
            recommend_result = model.docvecs.most_similar(user_rating_lst, topn=top_n)

            # 이때 데이터는 (레시피명,유사도) 튜플 형태로 반환된다
            # 추천된 레시피와 유사도 점수를 분리해서 담기

            mean_similarity_score = np.array([recommend_result[i][1] for i in range(len(recommend_result))]).mean()
            total_similarity_score.append(mean_similarity_score)
        except:
            total_similarity_score.append(None)

        similarity_score_df = pd.DataFrame(total_similarity_score)
        similarity_score_df.loc['average'] = similarity_score_df.mean()
        similarity_score_df.loc['std'] = np.sqrt(similarity_score_df.var())
        similarity_score_df.to_json(BASE_DIR + '/output/CBF_Recommender/CBF_Metrics')
        print('탐색이 완료되었습니다')


# %%
# 4.협업 필터링 추천 알고리즘
# %% 4-1. 훈련-데이터셋 분리
def CF1_spliting_train_test(ratings, TRAIN_SIZE=0.75):
    ratings = shuffle(ratings)
    cutoff = int(TRAIN_SIZE * len(ratings))
    ratings_train = ratings.iloc[:cutoff]
    ratings_test = ratings.iloc[cutoff:]

    return ratings_train, ratings_test


# %% 4-2.
def CF2_get_unseen_recipes(user_id):
    user_id = 20
    user_DB = pd.read_json(BASE_DIR + '/output/user_dummy_data')
    selected_recipe_names = user_DB['selected_recipe_name'].unique().tolist()
    selected_recipe_ranges = list(range(len(selected_recipe_names)))
    selected_recipes_dict = dict(zip(selected_recipe_names, selected_recipe_ranges))

    # 입력값으로 들어온 user_id에 해당하는 사용자가 평점을 매긴 모든 recipe를 리스트로 생성
    seen_recipes = user_DB[user_DB['user_id'] == user_id]['selected_recipe_name'].tolist()

    # 모든 recipe들의 recipe_id중 이미 평점을 매긴 recipe의 recipe_id를 제외하여 리스트로 생성
    unseen_recipes = [recipe for recipe in selected_recipe_names if recipe not in seen_recipes]
    unseen_recipes_id = [selected_recipes_dict[name] for name in unseen_recipes]
    print('평점 매긴 recipe 수:', len(seen_recipes), '추천 대상 recipe 수:', len(unseen_recipes), \
          '샘플 recipe 수:', len(selected_recipe_names))

    return unseen_recipes_id


# %% 4-4. 평가 척도: RMSE
def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))


# %% 4-R1 협업 필터링 적용
# 특정 유저의 좋아요 기록을 불러오기
def CF(user_id, model_loc=BASE_DIR + "/output/CF_Recommender/CF_Model.h5"):
    def RMSE(y_true, y_pred):
        return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

    # 200개로 추려진 요리 목록을 딕셔너리 형태로 담기
    ratings = pd.read_json(BASE_DIR + '/output/user_dummy_data')
    selected_recipe_names = ratings['selected_recipe_name'].unique().tolist()
    selected_recipe_ranges = list(range(len(selected_recipe_names)))
    selected_recipes_dict = dict(zip(selected_recipe_names, selected_recipe_ranges))

    # 유저들의 평가 데이터 불러오기
    # 그중 4점 이상 평가한 것으로 추린다
    user_rating_name = ratings[ratings['user_id'] == user_id]
    user_rating_name['stars'] = user_rating_name['stars'].apply(lambda x: int(x))
    user_rating_name = user_rating_name[user_rating_name['stars'] >= 4]
    user_rating_name = user_rating_name['selected_recipe_name']
    user_rating_lst = [selected_recipes_dict[name] for name in user_rating_name]

    # 모델 불러오기
    model = tf.keras.models.load_model(filepath=model_loc, custom_objects={'RMSE': RMSE})

    # 이미 평점을 메긴 정보를 제외하는 함수 불러오기
    unseen_recipes = CF2_get_unseen_recipes(user_id)

    # mu값을 구하기 위한 계산
    ratings_train, ratings_test = CF1_spliting_train_test(ratings)
    mu = ratings_train.stars.mean()  # 전체 평균

    # 레시피와 사용자 정보 배열로 만듬
    tmp_recipe_data = np.array(list(unseen_recipes))
    tmp_user = np.array([user_id for i in range(len(tmp_recipe_data))])

    # predict() list 객체로 저장
    predictions = model.predict([tmp_user, tmp_recipe_data])
    predictions = np.array([p[0] for p in predictions])

    # 정렬하여 인덱스 값 추출
    recommended_recipe_ids = (-predictions).argsort()[:top_n]
    top_recipe = recommended_recipe_ids

    recommend_result = top_recipe
    recommend_result = [selected_recipe_names[i] for i in recommend_result]

    # 이때 데이터는 (레시피명,유사도) 튜플 형태로 반환된다
    # 추천된 레시피와 유사도 점수를 분리해서 담기
    CF_df = pd.DataFrame([recommend_result, user_rating_name],
                         index=['recommended_recipe', 'user_preferred_recipe']).T

    CF_df.to_json(BASE_DIR + '/output/CF_Recommender/' + 'User_ID_' + str(user_id) + '_CF_results.json')


# %% 4-R2. 딥러닝 모델 설계 및 학습 & 저장
def make_CF_model():
    user_DB = pd.read_json(BASE_DIR + '/output/user_dummy_data')

    selected_recipe_names = user_DB['selected_recipe_name'].unique().tolist()
    selected_recipe_ranges = list(range(len(selected_recipe_names)))
    selected_recipes_dict = dict(zip(selected_recipe_names, selected_recipe_ranges))

    # 레시피 이름을 정수로 인코딩함
    user_DB['selected_recipe_id'] = [selected_recipes_dict[user_DB.iloc[i]['selected_recipe_name']] for i in
                                     range(len(user_DB))]
    ratings = user_DB
    ratings_train, ratings_test = CF1_spliting_train_test(user_DB)

    # Variable 초기화
    K = 100  # Latent factor 수
    mu = ratings_train.stars.mean()  # 전체 평균
    M = ratings.user_id.max() + 1  # Number of users
    N = ratings.selected_recipe_id.max() + 1  # Number of recipe

    # Keras model
    user = Input(shape=(1,))  # User input
    item = Input(shape=(1,))  # Item input
    P_embedding = Embedding(M, K, embeddings_regularizer=l2())(user)  # (M, 1, K)
    Q_embedding = Embedding(N, K, embeddings_regularizer=l2())(item)  # (N, 1, K)
    user_bias = Embedding(M, 1, embeddings_regularizer=l2())(user)  # User bias term (M, 1, )
    item_bias = Embedding(N, 1, embeddings_regularizer=l2())(item)  # Item bias term (N, 1, )

    # Concatenate layers
    P_embedding = Flatten()(P_embedding)  # (K, )
    Q_embedding = Flatten()(Q_embedding)  # (K, )
    user_bias = Flatten()(user_bias)  # (1, )
    item_bias = Flatten()(item_bias)  # (1, )
    R = Concatenate()([P_embedding, Q_embedding, user_bias, item_bias])  # (2K + 2, )

    # Neural network
    R = Dense(1024)(R)
    R = Activation('relu')(R)
    R = Dense(128)(R)
    R = Activation('linear')(R)
    R = Dense(1)(R)

    # 설계된 모델
    model = Model(inputs=[user, item], outputs=R)
    model.compile(
        loss=RMSE,  # 모델 학습 시 사용
        optimizer=SGD(),
        # optimizer=Adamax(),
        metrics=[RMSE]  # 모델 평가 시 사용
    )

    # 모델 학습
    result = model.fit(
        x=[ratings_train.user_id.values, ratings_train.selected_recipe_id.values],
        y=ratings_train.stars.values - mu,
        epochs=8,
        batch_size=512,
        validation_data=(
            [ratings_test.user_id.values, ratings_test.selected_recipe_id.values],
            ratings_test.stars.values - mu
        )
    )

    # Plot RMSE
    plt.plot(result.history['RMSE'], label="Train RMSE")
    plt.plot(result.history['val_RMSE'], label="Test RMSE")
    plt.xlabel('epoch')
    plt.ylabel('RMSE')
    plt.legend()
    plt.show()

    # 모델 저장
    filepath = BASE_DIR + "/output/CF_Recommender/CF_Model.h5"
    model.save(filepath)
    print('CF 모델이 업데이트 완료되었습니다')


# %%
def recommended_recipe_data_by_CBF(user_id):
    print('recommended_recipe_data_by_CBF: ', recommended_recipe_data_by_CBF)
    CBF(User_ID=user_id)
    print('user_id: ', user_id)
    print('BASE_DIR: ', BASE_DIR)
    recommender_df = pd.read_json(
        BASE_DIR + '/output/CBF_Recommender/' + 'User_ID_' + str(user_id) + '_CBF_results.json')
    print('recommender_df', recommender_df)
    recommended_recipe = list(recommender_df['recommended_recipe'])

    recipes = download_recipes()
    matched_recipes = pd.DataFrame()
    for recipe in recommended_recipe:
        matched_df = recipes[recipes['title'] == recipe]
        matched_recipes = pd.concat([matched_recipes, matched_df])
    matched_recipes.drop_duplicates(['title'], inplace=True)

    return matched_recipes


# %%
def recommended_recipe_data_by_CF(user_id):
    CF(user_id=user_id)
    recommender_df = pd.read_json(BASE_DIR + '/output/CF_Recommender/' + 'User_ID_' + str(user_id) + '_CF_results.json')
    recommended_recipe = list(recommender_df['recommended_recipe'])

    recipes = download_recipes()
    matched_recipes = pd.DataFrame()

    for recipe in recommended_recipe:
        matched_df = recipes[recipes['title'] == recipe]
        matched_recipes = pd.concat([matched_recipes, matched_df])
    matched_recipes.drop_duplicates(['title'], inplace=True)

    return matched_recipes


# %% 5. 필터링된 추천 알고리즘
# %%
def get_filter_data():
    # 인덱스가 숫자임에도 문자열로 인식되어 순서가 엉망이길래 정렬해줌
    def str2int(x):
        try:
            x = int(x)
            return x
        except:
            return x

    recipes_df = pd.read_json(BASE_DIR + '/output/Users_Filter.json')
    recipes_df.reset_index(inplace=True)
    recipes_df['index'] = recipes_df['index'].apply(lambda x: str2int(x))
    recipes_df.sort_values(by='index', inplace=True)
    recipes_df.set_index('index', inplace=True)

    title_lst = []
    for i in range(len(recipes_df)):
        title_lst.append(recipes_df.loc[i][0]['title'])

    return title_lst


# 두 리스트의 차집합을 구하는 함수
def difference_set(a, b):
    return list(set(a) - (set(b)))


# 두 리스트의 교집합을 구하는 함수
def intersection_set(a,b):
    return list(set(a) & set(b))


def filtered_CBF(user_id):
    filter_lst = get_filter_data()
    CBF(user_id)
    CBF_df = pd.read_json(BASE_DIR + '/output/CBF_Recommender/' + 'User_ID_' + str(user_id) + '_CBF_results.json')
    filtered_result = intersection_set(CBF_df['recommended_recipe'].tolist(), filter_lst)

    return filtered_result


def filtered_CF(user_id):
    filter_lst = get_filter_data()
    CF(user_id)
    CF_df = pd.read_json(BASE_DIR + '/output/CF_Recommender/' + 'User_ID_' + str(user_id) + '_CF_results.json')
    filtered_result = intersection_set(CF_df['recommended_recipe'].tolist(), filter_lst)

    return filtered_result


def filtered_recipe_data_by_CBF(user_id):
    recipes = download_recipes()
    filtered_result = filtered_CBF(user_id)

    matched_recipes = pd.DataFrame()
    for recipe in filtered_result:
        matched_df = recipes[recipes['title'] == recipe]
        matched_recipes = pd.concat([matched_recipes, matched_df])
    matched_recipes.drop_duplicates(['title'], inplace=True)
    return matched_recipes


def filtered_recipe_data_by_CF(user_id):
    recipes = download_recipes()
    filtered_result = filtered_CF(user_id)

    matched_recipes = pd.DataFrame()
    for recipe in filtered_result:
        matched_df = recipes[recipes['title'] == recipe]
        matched_recipes = pd.concat([matched_recipes, matched_df])
    matched_recipes.drop_duplicates(['title'], inplace=True)
    return matched_recipes
