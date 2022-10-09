from django.shortcuts import render, redirect
from django.contrib import auth
from django.core.exceptions import ValidationError

#맵제작에 필요한 모듈
from .models import MyBoard, MyMembers, MentalServiceLocation, Comment
import json
import folium
from folium.plugins import MarkerCluster
from folium import IFrame
from folium.plugins import MousePosition
from folium.plugins import FeatureGroupSubGroup
from folium.plugins import LocateControl
from folium import plugins

#기타
import re
from django.contrib.auth.hashers import check_password, make_password
import requests

#반경지도 모듈
import pandas as pd
from geopy.distance import great_circle
from jinja2 import Template

#크롤링
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

#위치검색에 사용되는 모듈
from geopy.geocoders import Nominatim

#페이지네이터
from django.core.paginator import Paginator,EmptyPage, PageNotAnInteger


def main_page(request):
    return render(request, 'main.html')


#병원데이터베이스 저장
def update_json(request):
    with open('전국_정신건강관련기관_위치정보.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # json에서 정보 불러오기
    for i in range(len(json_data['data'])):
        public_or_privates=(json_data['data'][i]['공공/민간'])
        categories=(json_data['data'][i]['기관구분'])
        agency=(json_data['data'][i]['기관명'])
        address=(json_data['data'][i]['주소'])
        latitude=(json_data['data'][i]['위도'])
        longitude=(json_data['data'][i]['경도'])
        phone_numbers=(json_data['data'][i]['전화번호'])
        introductions='안녕하세요. 자기소개글을 준비중입니다.'
        result = MentalServiceLocation.objects.create(public_or_privates=public_or_privates, categories=categories, agency= agency, address=address, latitude=latitude, longitude=longitude,phone_numbers=phone_numbers,introductions=introductions)
    return redirect('/')


## 마커로 위경도 가져옴.
class ClickForOneMarker(folium.ClickForMarker):
    _template = Template(u"""
            {% macro script(this, kwargs) %}
                var new_mark = L.marker();
                function newMarker(e){
                    new_mark.setLatLng(e.latlng).addTo({{this._parent.get_name()}});
                    new_mark.dragging.enable();
                    new_mark.on('dblclick', function(e){ {{this._parent.get_name()}}.removeLayer(e.target)})
                    var lat = e.latlng.lat.toFixed(4),
                       lng = e.latlng.lng.toFixed(4);
                    new_mark.bindPopup({{ this.popup }});
                    parent.document.getElementById("latitude").value = lat;
                    parent.document.getElementById("longitude").value =lng;
                    };
                {{this._parent.get_name()}}.on('click', newMarker);
            {% endmacro %}
            """)  # noqa

    def __init__(self, popup=None):
        super(ClickForOneMarker, self).__init__(popup)


##반경지도
class MappingByCoord:

    def __init__(self, df, lat, lng, dist=5):

        self.df = df
        self.lat = lat
        self.lng = lng
        self.dist = dist

    def setRectangler(self):

        lat_min = self.lat - 0.01 * self.dist
        lat_max = self.lat + 0.01 * self.dist

        lng_min = self.lng - 0.015 * self.dist
        lng_max = self.lng + 0.015 * self.dist

        self.points = [[lat_min, lng_min], [lat_max, lng_max]]

        result = self.df.loc[
            (self.df['latitude'] > lat_min) &
            (self.df['latitude'] < lat_max) &
            (self.df['longitude'] > lng_min) &
            (self.df['longitude'] < lng_max)
            ]
        result.index = range(len(result))

        return result

    def setCircle(self):

        tmp = self.setRectangler()

        center = (self.lat, self.lng)

        result = pd.DataFrame()

        for index, row in tmp.iterrows():
            point = (row['latitude'], row['longitude'])
            d = great_circle(center, point).kilometers
            if d <= self.dist:
                result = pd.concat([result, tmp.iloc[index, :].to_frame().T])

        result.index = range(len(result))

        return result

    def MappingInRectangler(self, df):
        m = folium.Map(location=[self.lat, self.lng], zoom_start=14)

        for idx, row in df.iterrows():
            lat_now = row['latitude']
            lng_now = row['longitude']

            folium.Marker(location=[lat_now, lng_now],
                          radius=15,
                          # tooltip=row['agency']).add_to(m)
                          popup=f"""<a href='{mental_agency(row['agency'])}'>{row['agency']}</h4><p align='center'>{row['address']}</p>""").add_to(m)

        folium.Rectangle(bounds=self.points,
                         color='#ff7800',
                         fill=True,
                         fill_color='#ffff00',
                         fill_opacity=0.2).add_to(m)

        return m

    def MappingInCircle(self,  df):

        ##줌 스타트 post방식으로 조절
        if self.dist <= 2:
            zoom = 14

        elif 3 < self.dist <= 5:
            zoom = 13

        elif 5 < self.dist <= 11:
            zoom = 12

        elif self.dist > 11:
            zoom = 11

        else:
            zoom = 11

        m = folium.Map(location=[self.lat, self.lng], width='100%', height='100%', zoom_start=zoom, tiles=None)
        folium.TileLayer('openstreetmap', name='구분').add_to(m)

        mcg = folium.plugins.MarkerCluster(control=False)
        m.add_child(mcg)
        sangdam = folium.plugins.FeatureGroupSubGroup(mcg, "상담소")
        center_ = folium.plugins.FeatureGroupSubGroup(mcg, "센터")
        ins = folium.plugins.FeatureGroupSubGroup(mcg, "시설")
        hos = folium.plugins.FeatureGroupSubGroup(mcg, "병원")
        bogun = folium.plugins.FeatureGroupSubGroup(mcg, "보건소")

        m.add_child(sangdam)
        m.add_child(center_)
        m.add_child(ins)
        m.add_child(hos)
        m.add_child(bogun)

        for idx, row in df.iterrows():

            lat_now = row['latitude']
            lng_now = row['longitude']

            div = ['상담소', '센터', '시설', '병원', '보건소']

            popup_html = f"""<a href="/mental_agency/{row['id']}" target='blank' style='font-weight:bold; font-size:12pt; color:#3f8edd;'>{row['agency']}</a><p>{row['address']}</p>"""
            popup_html = folium.Popup(popup_html, width=250, height=100, max_width=10000)

            if row['categories'] == div[0]:
                sangdam.add_child(
                    folium.Marker([lat_now, lng_now], icon=folium.Icon(color='pink'), radius=15, popup=popup_html))

            elif row['categories'] == div[1]:
                center_.add_child(
                    folium.Marker([lat_now, lng_now], icon=folium.Icon(color='green'), radius=15, popup=popup_html))

            elif row['categories'] == div[2]:
                ins.add_child(
                    folium.Marker([lat_now, lng_now], icon=folium.Icon(color='blue'), radius=15, popup=popup_html))

            elif row['categories'] == div[3]:
                hos.add_child(
                    folium.Marker([lat_now, lng_now], icon=folium.Icon(color='purple'), radius=15,popup=popup_html))

            else:
                bogun.add_child(
                    folium.Marker([lat_now, lng_now], icon=folium.Icon(color='orange'), radius=15, popup=popup_html))

        folium.Circle(radius=self.dist * 1000,
                      location=[self.lat, self.lng],
                      color="#ff7800",
                      fill_color='#ffff00',
                      fill_opacity=0.2
                      ).add_to(m)

        return m


def makeMap(request):
    request=request
    cfom = ClickForOneMarker()

    lat = request.POST['userLat']
    lng = request.POST['userLng']
    lat = float(lat.strip())
    lng = float(lng.strip())

    mylocation = [lat, lng]

    #dist_value = request.POST['dist']
    #dist = dist_value[0:2]
    #dist = float(dist.strip())

    dist = float(request.POST['dist'])

    df = pd.DataFrame(list(MentalServiceLocation.objects.all().values()))

    mbc = MappingByCoord(df, lat, lng, dist)

    result_radius = mbc.setCircle()

    mymap = mbc.MappingInCircle(result_radius)
    folium.Marker(location=mylocation, popup='현재 나의 위치', icon=folium.Icon(color='red', icon='star')).add_to(mymap)
    plugins.LocateControl().add_to(mymap)
    plugins.Geocoder(position='bottomright', collapsed=True, add_marker=True).add_to(mymap)

    folium.LayerControl(collapsed=True, position='bottomright').add_to(mymap)
    mymap.add_child(cfom)

    mymap.layer_name = '구분'

    maps = mymap._repr_html_()

    ##지도 정보 가져와서 리스트 만들기
    df2 = pd.DataFrame(list(MentalServiceLocation.objects.all().values()))

    lat_min = lat - 0.01 * dist
    lat_max = lat + 0.01 * dist
    lng_min = lng - 0.015 * dist
    lng_max = lng + 0.015 * dist
    points = [[lat_min, lng_min], [lat_max, lng_max]]
    result = df2.loc[
        (df2['latitude'] > lat_min) &
        (df2['latitude'] < lat_max) &
        (df2['longitude'] > lng_min) &
        (df2['longitude'] < lng_max)
        ]
    result.index = range(len(result))
    tmp = result
    center = (lat, lng)
    result_circle = pd.DataFrame()
    for index, row in tmp.iterrows():
        point = (row['latitude'], row['longitude'])
        d = great_circle(center, point).kilometers
        if d <= dist:
            result_circle = pd.concat([result_circle, tmp.iloc[index, :].to_frame().T])
    result_circle.index = range(len(result_circle))
    result_json = result_circle.to_json(orient='records', force_ascii=False)
    lists = json.loads(result_json)
    return render(request, 'map_show.html', {'mymap': maps, 'lists': lists})


##주소로 검색
def makeMap_by_address(request):
    cfom = ClickForOneMarker()
    #입력된 위치정보를 좌표값으로 변환하기
    address = request.POST['address']

    api_key = 'a1fe3f09ee0f56aa05558e8efc6db52e'
    url = 'https://dapi.kakao.com/v2/local/search/address.json?query={address}'.format(address=address)
    headers = {"Authorization": "KakaoAK " + api_key}
    try:
        result = json.loads(str(requests.get(url, headers=headers).text))
        match_first = result['documents'][0]['address']
        # 위도,경도 반환
        mylocation = [float(match_first['y']), float(match_first['x'])]
        lat = mylocation[0]
        lng = mylocation[1]
        dist = float(request.POST['dist'])

        df = pd.DataFrame(list(MentalServiceLocation.objects.all().values()))

        mbc = MappingByCoord(df, lat, lng, dist)

        result_radius = mbc.setCircle()

        mymap = mbc.MappingInCircle(result_radius)
        folium.Marker(location=mylocation, popup='현재 나의 위치', icon=folium.Icon(color='red', icon='star')).add_to(
            mymap)
        plugins.LocateControl().add_to(mymap)
        plugins.Geocoder(position='bottomright', collapsed=True, add_marker=True).add_to(mymap)

        folium.LayerControl(collapsed=True, position='bottomright').add_to(mymap)
        mymap.add_child(cfom)

        mymap.layer_name = '구분'

        maps = mymap._repr_html_()

        ##지도 정보 가져와서 리스트 만들기
        df2 = pd.DataFrame(list(MentalServiceLocation.objects.all().values()))

        lat_min = lat - 0.01 * dist
        lat_max = lat + 0.01 * dist
        lng_min = lng - 0.015 * dist
        lng_max = lng + 0.015 * dist
        points = [[lat_min, lng_min], [lat_max, lng_max]]
        result = df2.loc[
            (df2['latitude'] > lat_min) &
            (df2['latitude'] < lat_max) &
            (df2['longitude'] > lng_min) &
            (df2['longitude'] < lng_max)
            ]
        result.index = range(len(result))
        tmp = result
        center = (lat, lng)
        result_circle = pd.DataFrame()
        for index, row in tmp.iterrows():
            point = (row['latitude'], row['longitude'])
            d = great_circle(center, point).kilometers
            if d <= dist:
                result_circle = pd.concat([result_circle, tmp.iloc[index, :].to_frame().T])
        result_circle.index = range(len(result_circle))
        result_json = result_circle.to_json(orient='records', force_ascii=False)
        lists = json.loads(result_json)

        return render(request, 'map_show.html', {'mymap': maps, 'lists': lists})

    except:
        return redirect('/')


##전체지도
def make_map(request,my_location=[37.5, 126.98],radius=5000):
    # #기능1. 장고데이터베이스에서 정보 가져오기
    # #상담소/공공/민간데이터 정보로 구분하기
    # counseling_data = MentalServiceLocation.objects.filter(categories='상담소')
    # center_data = MentalServiceLocation.objects.filter(categories='센터')
    # ins_data = MentalServiceLocation.objects.filter(categories='시설')
    # hos_data = MentalServiceLocation.objects.filter(categories='병원')
    # ph_data = MentalServiceLocation.objects.filter(categories='보건소')
    # #상담소/센터/시설/병원/보건소
    # #상담소
    # counseling_agency=[]
    # counseling_address=[]
    # counseling_latitude=[]
    # counseling_longitude=[]
    # counseling_phone_numbers=[]
    # #센터
    # center_agency=[]
    # center_address=[]
    # center_latitude=[]
    # center_longitude=[]
    # center_phone_numbers=[]
    # #시설
    # ins_agency=[]
    # ins_address=[]
    # ins_latitude=[]
    # ins_longitude=[]
    # ins_phone_numbers=[]
    # # 병원
    # hos_agency = []
    # hos_address = []
    # hos_latitude = []
    # hos_longitude = []
    # hos_phone_numbers = []
    # # 보건소
    # ph_agency = []
    # ph_address = []
    # ph_latitude = []
    # ph_longitude = []
    # ph_phone_numbers = []
    # #데이터 삽입
    # for c_data in counseling_data:
    #     counseling_agency.append(c_data.agency)
    #     counseling_address.append(c_data.address)
    #     counseling_latitude.append(c_data.latitude)
    #     counseling_longitude.append(c_data.longitude)
    #     counseling_phone_numbers.append(c_data.phone_numbers)
    # for ct_data in center_data:
    #     center_agency.append(ct_data.agency)
    #     center_address.append(ct_data.address)
    #     center_latitude.append(ct_data.latitude)
    #     center_longitude.append(ct_data.longitude)
    #     center_phone_numbers.append(ct_data.phone_numbers)
    # for i_data in ins_data:
    #     ins_agency.append(i_data.agency)
    #     ins_address.append(i_data.address)
    #     ins_latitude.append(i_data.latitude)
    #     ins_longitude.append(i_data.longitude)
    #     ins_phone_numbers.append(i_data.phone_numbers)
    # for h_data in hos_data:
    #     hos_agency.append(h_data.agency)
    #     hos_address.append(h_data.address)
    #     hos_latitude.append(h_data.latitude)
    #     hos_longitude.append(h_data.longitude)
    #     hos_phone_numbers.append(h_data.phone_numbers)
    # for p_data in ph_data:
    #     ph_agency.append(p_data.agency)
    #     ph_address.append(p_data.address)
    #     ph_latitude.append(p_data.latitude)
    #     ph_longitude.append(p_data.longitude)
    #     ph_phone_numbers.append(p_data.phone_numbers)
    #지도 만들기
    my_map = folium.Map(location=my_location, zoom_start=12.0,
                        tiles=None, control_scale=True)
    folium.TileLayer('openstreetmap', name='구분').add_to(my_map)

    # ##선택 카테고리 추가하기
    # mcg = folium.plugins.MarkerCluster(control=False)
    # my_map.add_child(mcg)
    # sangdam = folium.plugins.FeatureGroupSubGroup(mcg, "상담소")
    # center_ = folium.plugins.FeatureGroupSubGroup(mcg, "센터")
    # ins = folium.plugins.FeatureGroupSubGroup(mcg, "시설")
    # hos = folium.plugins.FeatureGroupSubGroup(mcg, "병원")
    # bogun = folium.plugins.FeatureGroupSubGroup(mcg, "보건소")
    # my_map.add_child(sangdam)
    # my_map.add_child(center_)
    # my_map.add_child(ins)
    # my_map.add_child(hos)
    # my_map.add_child(bogun)

    # # 상담소 마커
    # for name, lat, long, address_ in zip(counseling_agency,counseling_latitude,counseling_longitude,counseling_address):
    #     text=f"""<h4 align='center'>{name}</h4><p align='center'>{address_}</p>"""
    #     iframe = folium.IFrame(text, width=100, height=100)
    #     popup = folium.Popup(iframe, max_width=3000)
    #     sangdam.add_child(
    #         folium.Marker([lat, long], popup = popup, icon=folium.Icon(color='pink')))
    # #센터 마커
    # for name, lat, long, address_ in zip(center_agency,center_latitude,center_longitude,center_address):
    #     text = f"""<h4 align='center'>{name}</h4><p align='center'>{address_}</p>"""
    #     iframe= folium.IFrame(text, width=100, height=100)
    #     popup = folium.Popup(iframe, max_width=3000)
    #     center_.add_child(
    #         folium.Marker([lat, long], popup = popup, icon=folium.Icon(color='green')))
    # #시설 마커
    # for name, lat, long, address_ in zip(ins_agency,ins_latitude,ins_longitude,ins_address):
    #     text=f"""<h4 align='center'>{name}</h4><p align='center'>{address_}</p>"""
    #     iframe = folium.IFrame(text, width=100, height=100)
    #     popup = folium.Popup(iframe, max_width=3000)
    #     ins.add_child(
    #         folium.Marker([lat, long], popup = popup, icon=folium.Icon(color='blue')))
    # #병원 마커
    # for name, lat, long, address_ in zip(hos_agency,hos_latitude,hos_longitude,hos_address):
    #     text=f"""<h4 align='center'>{name}</h4><p align='center'>{address_}</p>"""
    #     iframe = folium.IFrame(text, width=100, height=100)
    #     popup = folium.Popup(iframe, max_width=3000)
    #     hos.add_child(
    #         folium.Marker([lat, long], popup = popup, icon=folium.Icon(color='purple')))
    # #보건소 마커
    # for name, lat, long, address_ in zip(ph_agency,ph_latitude,ph_longitude,ph_address):
    #     text=f"""<h4 align='center'>{name}</h4><p align='center'>{address_}</p>"""
    #     iframe = folium.IFrame(text, width=100, height=100)
    #     popup = folium.Popup(iframe, max_width=3000)
    #     bogun.add_child(
    #         folium.Marker([lat, long], popup = popup, icon=folium.Icon(color='orange')))
    # LocateControl().add_to(my_map)
    # plugins.Geocoder(position='bottomright', collapsed=True, add_marker=True).add_to(my_map)
    # folium.LayerControl(position='bottomright', collapsed=True).add_to(my_map)

    maps = my_map._repr_html_()

    return render(request, 'map_show.html', {'mymap': maps})

def find_markers(request):
    # 페이지에 생성된 모든 마커들을 크롤링하여 위치정보를 찾아내기
    service = Service(executable_path=ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service)
    url = 'http://127.0.0.1:8000/map_show/'
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    marker_sources= str(soup.select("div > div > div"))
    marker_sources2=re.findall('\[.+?\]',marker_sources)
    remove_set=['[0]']
    markers = [i for i in marker_sources2 if i not in remove_set][1:]
    for i in range(len(markers)):
        markers[i] = markers[i].strip('[').strip(']').split(',')
        for j in range(len(markers[i])):
            markers[i][j] =float(markers[i][j])


    # 데이터베이스에서 정보 불러오기
    json_datas = MentalServiceLocation.objects.all()
    agency=[]
    categories=[]
    address=[]
    latitude=[]
    longitude=[]
    phone_numbers=[]
    json_latitude=[]
    json_longitude=[]

    for json_data in json_datas:
        for marker in markers:
            json_latitude=float(str(json_data.latitude).strip('Decimal').strip('(').strip(')').strip('\''))
            json_longitude=float(str(json_data.longitude).strip('Decimal').strip('(').strip(')').strip('\''))
            if [json_latitude,json_longitude] == marker:

                agency.append(json_data.agency)
                categories.append(json_data.categories)
                address.append(json_data.address)

    return redirect('/map_show/')


## 게시판
def mental_agency(request, id):
    if request.method == 'GET':
        # agency 정보
        mental_agency = MentalServiceLocation.objects.get(id=id)
        # 해당 agency board list 정보
        # 지정된 agency에서 session에 저장된 사용자의 정보만 가져오기.
        try:
            list = MyBoard.objects.filter(mental_address = id, members_mynickname = request.session['mynickname'])

        except KeyError:
            return redirect('login')

        context = { 'mental_agency' : mental_agency ,'mental_board_myList': list}
        return render(request, 'mental_agency.html', context)

    else:
        # 해당 agency board create
        mytitle = request.POST['board_title']
        mycontent = request.POST['board_content']

        mental_agency = MentalServiceLocation.objects.get(id=id)

        mymember = MyMembers.objects.get(mynickname = request.session['mynickname'])
        result = MyBoard.objects.create(members_mynickname= mymember, mental_address=mental_agency, board_title=mytitle ,board_content=mycontent)

        if result:
            return redirect('/mental_agency/'+str(id))
        else:
            return redirect('/mental_agency/'+str(id))


def mynick_boardlist(request):
    if request.method == 'GET':
        myboard_all = MyBoard.objects.filter(members_mynickname=request.session['mynickname']).order_by('-id')

        paginator = Paginator(myboard_all, 5)
        page_num = request.GET.get('page', '1')

        page_obj = paginator.get_page(page_num)

        return render(request, 'mynick_boardlist.html', {'list': page_obj})
    else:
        return


## 일반 사용자 회원가입 및 로그인, 로그아웃
def register(request):
    if request.method == 'GET':
        return render(request, 'register.html')
    else:
        if MyMembers.objects.filter(mynickname=request.POST['mynickname']).exists():

            return redirect('register')

        else:
            myemail = request.POST['myemail']
            mypassword = request.POST['mypassword']
            mynickname = request.POST['mynickname']
            mymember = MyMembers(myemail=myemail, mypassword=make_password(mypassword), mynickname=mynickname)
            mymember.save()

            return redirect('login')


def login(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    else:
        mynickname = request.POST['mynickname']
        mypassword = request.POST['mypassword']
        try:
            mymember = MyMembers.objects.get(mynickname=mynickname)

            if not check_password(mypassword, mymember.mypassword):
                return redirect('/login/')

            else:
                request.session['mynickname'] = mymember.mynickname
                return redirect('/map_show/')

        except MyMembers.DoesNotExist:
            return redirect('/login/')



def logout(request):
    del request.session['mynickname']
    return redirect('/map_show/')


# agency : main (board list) & 회원가입 및 로그인 로그아웃
def agency_main(request):
    agency_name = MentalServiceLocation.objects.get(agency_email = request.session['agency_email'])
    agency_id = MentalServiceLocation.objects.get(agency_email = request.session['agency_email']).id
    agency_board_list = MyBoard.objects.filter(mental_address=agency_id)

    content = {'list' : agency_board_list , 'agency' : agency_name}

    return render(request, 'agency_main.html', content )


def agency_comment(request, id):
    if request.method == 'GET':
        # agency 이름 가져오기 위해서
        agency = MentalServiceLocation.objects.get(agency_email = request.session['agency_email'])

        # board랑 관련 comment 가져오기
        agency_board = MyBoard.objects.prefetch_related('comment_set').filter(id=id)

        content = {'dto' : agency_board, 'agency_data' : agency }
        return render(request, 'agency_comment.html', content)
    else:
        reply = request.POST['comment_reply']
        comment = Comment(reply=reply, post_id=id)
        comment.save()
        return redirect('agency_main')


def agency_register(request):
    if request.method == 'GET':
        return render(request, 'agency_register.html')
    else:
        agency_id = request.POST['agency_ID']
        agency_email = request.POST['agency_email']
        agency_password = request.POST['agency_password']

        agency_mymember = MentalServiceLocation.objects.filter(id=agency_id)

        agency_member = agency_mymember.update(agency_email = agency_email, agency_password = make_password(agency_password))

        return redirect('agency_login')


def agency_login(request):
    if request.method == 'GET':
        return render(request, 'agency_login.html')
    else:
        agency_email = request.POST['agency_email']
        agency_password = request.POST['agency_password']
        agency_member = MentalServiceLocation.objects.filter(agency_email=agency_email).first()
        # print(agency_member.id) = 196

        if not check_password(agency_password, agency_member.agency_password):
            request.session['agency_email'] = agency_member.agency_email
            raise ValidationError("패스워드가 틀립니다.")
        else:
            request.session['agency_email'] = agency_member.agency_email
            return redirect('/agency_main')

def agency_logout(request):
    del request.session['agency_email']
    return redirect('/')
