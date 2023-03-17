"""pjt3_vegan_recipes URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib.auth.views import LoginView, LogoutView
from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('recipe/<int:id>', views.recipe, name='recipe'),
    path('signup_1/', views.signup_1, name='signup_1'),
    path('signup_2/', views.signup_2, name='signup_2'),
    path('signup_3/', views.signup_3, name='signup_3'),
    path('signup_4/', views.signup_4, name='signup_4'),
    path('signup_5/', views.signup_5, name='signup_5'),
    path('', views.main, name='main'),
    path('main_login/', views.recommend_by_algorithm, name='recommend_by_algorithm'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),
    path('pinned_recipe/', views.pinned_recipe, name='pinned_recipe'),
    path('pin_recipe/<int:id>', views.pin_recipe, name='pin_recipe'),
    path('search_result/', views.search_result, name='search_result'),
    path('search_result_q/', views.search_result_q, name='search_result_q'),
    path('about_us/', views.about_us, name='about_us'),
    path('algorithm/', views.algorithm, name='algorithm'),
    path('rate/<int:id>', views.rate, name='rate'),
    path('signup_rate_1/<int:id>', views.signup_rate_1, name='signup_rate_1'),
    path('signup_rate_2/<int:id>', views.signup_rate_2, name='signup_rate_2'),
    path('signup_rate_3/<int:id>', views.signup_rate_3, name='signup_rate_3'),
    path('signup_rate_4/<int:id>', views.signup_rate_4, name='signup_rate_4'),
    # 알고리즘 작동 확인용
    path('show_CBF/', views.show_CBF, name='show_CBF'),
    path('show_CF/', views.show_CF, name='show_CF'),
    path('show_rating/', views.show_rating, name='show_rating'),
    # 모델 업데이트 및 더미데이터 제작
    path('update_cluster/', views.update_cluster, name='update_cluster'),
    path('update_CBF/', views.update_CBF, name='update_CBF'),
    path('update_CF/', views.update_CF, name='update_CF'),
    path('make_dummy/', views.make_dummy, name='make_dummy'),
    path('recommend_by_algorithm/', views.recommend_by_algorithm, name='recommend_by_algorithm'),
    path('main_login_q/', views.main_login_q, name='main_login_q'),
]
