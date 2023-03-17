"""visualization URL Configuration

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
from xml.etree.ElementInclude import include
from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.main_page, name='main_page'),
    path('map_show/', views.make_map, name='map_show'),
    path('radius/', views.makeMap, name='radius'),
    path('mental_service_location/', views.update_json, name='mental_service_location'),
    path('makeMap_by_address/', views.makeMap_by_address, name='makeMap_by_address'),

    path('mental_agency/<int:id>', views.mental_agency, name='mental_agency'),

    # 일반 사용자
    path('mynick_boardlist/', views.mynick_boardlist, name='mynickboardlist'),
    path('register/', views.register, name='register'),
    path('login/', views.login, name='login'),
    path('logout/', views.logout, name='logout'),

    # mental_agency 관계자
    path('agency_login/', views.agency_login, name='agency_login'),
    path('agency_logout/', views.agency_logout, name='agency_logout'),
    path('agency_register/', views.agency_register, name='agency_register'),
    path('agency_main/', views.agency_main, name='agency_main'),
    path('agency_comment/<int:id>', views.agency_comment, name='agency_comment'),
]
