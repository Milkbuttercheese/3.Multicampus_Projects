from django import forms
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User

class MyMemberForm(UserCreationForm):
    """
    UserCreationForm에는 기본적으로 'username', 'password1', 'password2' field가 존재함!
    'password1' : 비밀번호
    'password2' : 비밀번호 확인
    """
    email = forms.EmailField()
    first_name = forms.CharField()
    last_name = forms.CharField()

    class Meta:
        model = User
        fields = ('username', 'password1', 'password2', 'email', 'first_name', 'last_name')