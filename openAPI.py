### 오픈 API 데이터 받아오기 ###
import pandas as pd
from bs4 import BeautifulSoup
import urllib.request as req
from urllib.parse import urlencode, quote_plus


## URL request --> 받아오기 ##
## https://www.juso.go.kr/ ## open API

http://www.juso.go.kr/addrlink/addrLinkApi.do?currentPage=1&countPerPage=10&keywo
rd=강서로7길&confmKey=TESTJUSOGOKR



url = 'http://www.juso.go.kr/addrlink/addrLinkApi.do?currentPage=1&countPerPage=10' + '&'
keyword = "keyword=" + address 
confmKey = 'confmKey=' + "U01TX0FVVEgyMDE4MTAzMTE0MTEzNzEwODI3MDA=" # 컨펌키 = 서비스키 (오픈 API)
url_all = url + confmKey
html = req.urlopen(url_all).read().decode('utf-8')
soup = BeautifulSoup(html, 'lxml-xml', from_encoding='utf-8')
price = soup.select_one("item")
print(price)