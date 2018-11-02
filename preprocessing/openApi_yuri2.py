### 오픈 API 데이터 받아오기 ###
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.request as req
from urllib.parse import urlencode, quote_plus

code = {'종로구': '11110', '중구': '11140', '용산구': '11170', '성동구': '11200', 
        '광진구': '11215', '동대문구': '11230', '중랑구': '11260', '성북구': '11290', 
        '강북구': '11305', '도봉구': '11320', '노원구': '11350', '은평구': '11380', 
        '서대문구': '11410', '마포구': '11440', '양천구': '11470', '강서구': '11500',
        '구로구': '11530', '금천구': '11545', '영등포구': '11560', '동작구': '11590',
        '관악구': '11620', '서초구': '11650', '강남구': '11680', '송파구': '11710', '강동구': '11740'}

###############################
key_list = []
values_list = []
for key in code.keys():
    if key not in key_list:
        key_list.append(key)
    else:
        pass
print(key_list)
cnt = 0
for value in code.values():
    if value not in values_list:
        values_list.append(value)
        cnt+=1
    else:
        pass
print(values_list)
# print(cnt)

## URL request --> 받아오기 ## --> 하루 1000트래픽 한정(1 계정당)
url = 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade?'
# 서비스키 중요 --> 공공데이터포털에서 오픈API로 받은 인증키를 여기에 입력
serviceKey = 'serviceKey=' + "9rL%2Fv5RbvEC4bLji%2B3hTKF1wuBPkoVk6nnKg54x54CON8OOz5cLFmRFdIWf38wJmwxilBt4XYpuqm0jzpCoM8w%3D%3D" + "&"
LAWD_CD = 'LAWD_CD='
DEAL_YMD = 'DEAL_YMD='
########################
# LAWD_CD = 'LAWD_CD=' + code['종로구'] + '&'  # 법정 코드 번호 --> 가운데 숫자만 변화주면됨. (위 codedict)
# DEAL_YMD = 'DEAL_YMD=' + "201801"  # 기간 --> 수집시기는 우리의 몫
# url_all = url + serviceKey + LAWD_CD + DEAL_YMD
########################

value_tmplist=[]
for v in values_list:
    LAWD_CD = 'LAWD_CD='
    if 'LAWD_CD='+v+'&' not in value_tmplist:
        LAWD_CD += v + '&'
        value_tmplist.append(LAWD_CD)
    else:
        continue
# print(value_tmplist)

date_tmplist=[]
for i in range(2015,2019):
    DEAL_YMD = 'DEAL_YMD='
    for k in range(1,13):
        if len(str(k)) != 2 :
            day ='0'+str(k)
        else:
            day = str(k)
        if 'DEAL_YMD=' + str(i) + day not in date_tmplist:
            DEAL_YMD = 'DEAL_YMD=' + str(i) + day
            date_tmplist.append(DEAL_YMD)
        else:
            continue
# print(len(date_tmplist))

url_tmplist =[]
url_alltmplist = []
for i in range(len(key_list)):
    url_all = url + serviceKey
    for k in range(len(value_tmplist)):
        if url_all + value_tmplist[k] not in url_tmplist :
            url_all2 = url_all + value_tmplist[k]
            url_tmplist.append(url_all2)
        else:
            continue
        for m in range(len(date_tmplist)):
            if url_tmplist[k] + date_tmplist[m] not in url_alltmplist:
                url_all3 = url_tmplist[k] + date_tmplist[m]
                url_alltmplist.append(url_all3)
            else:
                break

print(url_tmplist)
print(url_alltmplist)
########################
print(len(url_alltmplist))
########################


html = req.urlopen(url_all).read().decode('utf-8')
soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')
price = soup.select_one("item")
print(price)

### 변수명 ###
Apt = ""
Add = ""
Price = 1
Area = 1
per_Pyeong = Price / Area

'''
실거래가 DateFrame을 만드는 게 목적 [소득 추정의 설명력을 높이기 위한 변수 추가]
     Apt(아파트) |  Add(동+지번) |  Price(거래가) |  Area(전용면적)  |  per_Pyeong(평당 가격)  |  층  |
---------------------------------------------------------------------------------------------------
        래미안   |   사직동 21   |     78,100    |       29.76      |      78100/29.76       |  4   |
이 Df에도 --> "좌표값 추가"
'''
