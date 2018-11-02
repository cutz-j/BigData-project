### 오픈 API 데이터 받아오기 ### --> 도로명- 구주소 변환 
import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np

## 어린이집 데이터 전처리 ##
all_center = pd.read_csv("d:/project_data/total_school01.csv", sep=",", encoding="euc-kr")
for i in range(len(all_center['Add'])): # 주소 분리 후 주소 + 숫자로
    tmpList = all_center['Add'][i].split()
    all_center["Add"][i] = " ".join([tmpList[2], tmpList[3]])

# 국공립만 선택 #
x_test = all_center[all_center['Type'] == "국공립"] # 국공립만 선택
all_center['old_add'] = 0

## URL request --> 받아오기 ##
## https://www.juso.go.kr/ ## open API
## 주소 형식 ##
# http://www.juso.go.kr/addrlink/addrLinkApi.do?currentPage=1&countPerPage=10&keyword=%EC%82%BC%EC%B2%AD%EB%A1%9C7%EA%B8%B826&confmKey=U01TX0FVVEgyMDE4MTAzMTE0MTEzNzEwODI3MDA=

## parse ##

address = all_center['Add'].values # 어린이집 새주소 뽑기
url = 'http://www.juso.go.kr/addrlink/addrLinkApi.do?currentPage=1&countPerPage=10' + '&'
confmKey = 'confmKey=' + "U01TX0FVVEgyMDE4MTAzMTE0MTEzNzEwODI3MDA=" # 컨펌키 = 서비스키 (오픈 API)
for i in range(len(address)):
    word = address[i]
#    print(word)
    keyword = "keyword=" + word.replace(" ", "") + '&'
    url_all = url + keyword + confmKey # url 합치기
    req = requests.get(url_all) # req
    html = req.text
    soup = BeautifulSoup(html, 'lxml-xml', from_encoding='utf-8')
    try:
        add = soup.select_one("jibunAddr").text # 구주소 뽑아오기
        zipnum = soup.select_one("zipNo").text
    except: continue
    split = add.split()
    all_center['old_add'][i] = split[2] + " " + split[3]
    all_center['Code'][i] = zipnum

all_center.to_csv("d:/project_data/all_data2.csv", encoding='euc-kr')
