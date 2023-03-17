from django.core.exceptions import ValidationError
from django.shortcuts import render, redirect
from django.contrib import auth

# 일반 사용자
def validemail(request):
    if not '@' in request.POST['myemail']:
        raise ValidationError('이메일 형식이 아닙니다.')

    else:
        return request


def validpw(request):
    pwval = request.POST['mypassword']

    if len(pwval) < 8:
        raise ValidationError("패스워드가 너무 짧습니다")
    elif len(pwval) > 20:
        raise ValidationError("패스워드가 너무 깁니다")
    else:
        return request


# 관리자용 사용자
def agency_validemail(request):
    if not '@' in request.POST['agency_email']:
        raise ValidationError('이메일 형식이 아닙니다.')

    else:
        return request


def agency_validpw(request):
    agencypw_val = request.POST['agency_password']

    if len(agencypw_val) < 8:
        raise ValidationError("패스워드가 너무 짧습니다")
    elif len(agencypw_val) > 20:
        raise ValidationError("패스워드가 너무 깁니다")
    else:
        return request