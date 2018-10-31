### 오픈 API 데이터 받아오기 ### --> 도로명- 구주소 변환 
import pandas as pd
from bs4 import BeautifulSoup
import requests
import numpy as np
import json

## 어린이집 데이터 전처리 ##
all_center = json.load(open("d:/project_data/allmap.json", encoding='utf-8'))
c_header = all_center['DESCRIPTION'] # JSON 분리
c_data = all_center['DATA']
c_alldf = pd.DataFrame(c_data)

# 특정 열만 선택 #
c_alldf = c_alldf[['cot_conts_name', 'cot_coord_x', 'cot_coord_y', 'cot_value_01', 'cot_gu_name','cot_addr_full_new']]
c_alldf.columns = ['name', 'x', 'y', 'kinds', 'location','address']
x_test = c_alldf[c_alldf['kinds'] == "국공립"] # 국공립만 선택

## URL request --> 받아오기 ##
## https://www.juso.go.kr/ ## open API
## 주소 형식 ##
# http://www.juso.go.kr/addrlink/addrLinkApi.do?currentPage=1&countPerPage=10&keyword=%EC%82%BC%EC%B2%AD%EB%A1%9C7%EA%B8%B826&confmKey=U01TX0FVVEgyMDE4MTAzMTE0MTEzNzEwODI3MDA=

## parse ##
address = np.array(x_test['address']) # 어린이집 새주소 뽑기
url = 'http://www.juso.go.kr/addrlink/addrLinkApi.do?currentPage=1&countPerPage=10' + '&'
keyword = "keyword=" + address[1].replace(" ", "") + '&'
confmKey = 'confmKey=' + "U01TX0FVVEgyMDE4MTAzMTE0MTEzNzEwODI3MDA=" # 컨펌키 = 서비스키 (오픈 API)
url_all = url + keyword + confmKey # url 합치기
req = requests.get(url_all) # req
html = req.text
soup = BeautifulSoup(html, 'lxml-xml', from_encoding='utf-8')
add = soup.select_one("jibunAddr").text # 구주소 뽑아오기
print(add.split()) # split