#!/usr/bin/env python
# coding: utf-8

# In[2]:


### 오픈 API 데이터 받아오기 ###
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.request as req
from urllib.parse import urlencode, quote_plus

### 만들어야할 코드 내용 요약 ###
# LAWD_CD = 'LAWD_CD=' + code['종로구'] + '&'  # 법정 코드 번호 --> 가운데 숫자만 변화주면됨. (위 codedict)
# DEAL_YMD = 'DEAL_YMD=' + "201801"  # 기간 --> 수집시기는 우리의 몫
# url_all = url + serviceKey + LAWD_CD + DEAL_YMD
########################################################################################################################


### 아파트 url 생성 코드 작성 ###
# URL request --> 받아오기 ## --> 하루 1000트래픽 한정(1 계정당)
# 서비스키 중요 --> 공공데이터포털에서 오픈API로 받은 인증키를 여기에 입력
url = 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcRHTrade?'
serviceKey = 'serviceKey=' + "0d8fGluCLeDwmtW310ls9LnNRS582k2fwYEnmtr25HJ8Iv%2Bwcjd4D%2B6M4wQNwuCgLTrDHSawkREI6gD0uHlYGA%3D%3D" + "&"
LAWD_CD = 'LAWD_CD='
DEAL_YMD = 'DEAL_YMD='

### 연립다세대 url 생성 코드 작성 ###


### 지역코드 생성 ###

code = {'종로구': '11110', '중구': '11140', '용산구': '11170', '성동구': '11200',
        '광진구': '11215', '동대문구': '11230', '중랑구': '11260', '성북구': '11290', 
        '강북구': '11305', '도봉구': '11320', '노원구': '11350', '은평구': '11380', 
        '서대문구': '11410', '마포구': '11440', '양천구': '11470', '강서구': '11500',
        '구로구': '11530', '금천구': '11545', '영등포구': '11560', '동작구': '11590',
        '관악구': '11620', '서초구': '11650', '강남구': '11680', '송파구': '11710', '강동구': '11740'}

values_list = []
cnt = 0
for value in code.values():
    if value not in values_list:
        values_list.append(value)
        cnt+=1
    else:
        pass
# print(values_list)

value_tmplist=[]
for v in values_list:
    LAWD_CD = 'LAWD_CD='
    if 'LAWD_CD='+v+'&' not in value_tmplist:
        LAWD_CD += v + '&'
        value_tmplist.append(LAWD_CD)
    else:
        continue
# print(value_tmplist)...  'LAWD_CD=' + code['종로구'] + '&'


### 검색기간(Monthly) 생성 ###

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
# print(date_tmplist)   ... 'DEAL_YMD=' + "201801"


### URL 생성(합치기) ###

url_tmplist =[]
url_all_list = []
for i in range(len(values_list)):
    url_all = url + serviceKey
    for k in range(len(value_tmplist)):
        if url_all + value_tmplist[k] not in url_tmplist :
            url_all2 = url_all + value_tmplist[k]
            url_tmplist.append(url_all2)
        else:
            continue
        for m in range(len(date_tmplist)):
            if url_tmplist[k] + date_tmplist[m] not in url_all_list:
                url_all3 = url_tmplist[k] + date_tmplist[m]
                url_all_list.append(url_all3)
            else:
                break

print(url_all_list)
print(len(url_all_list))


### URL연동 데이터 파싱하기 (미완) ###

html = req.urlopen(url_all).read().decode('utf-8')
soup = BeautifulSoup(html, 'html.parser', from_encoding='utf-8')
price = soup.select_one("item")
print(price)


### DataFrame에 사용할 열이름(Columns) ###
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

### DataFrame 생성하기 ###
bilaDf = pd.DataFrame(np.arange())

# In[ ]:




