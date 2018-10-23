### 오픈 API 데이터 받아오기 ###
import pandas as pd
from bs4 import BeautifulSoup
import urllib.request as req
from urllib.parse import urlencode, quote_plus

code = {'종로구': '11110', '중구': '11140', '용산구': '11170', '성동구': '11200', 
        '광진구': '11215', '동대문구': '11230', '중랑구': '11260', '성북구': '11290', 
        '강북구': '11305', '도봉구': '11320', '노원구': '11350', '은평구': '11380', 
        '서대문구': '11410', '마포구': '11440', '양천구': '11470', '강서구': '11500',
        '구로구': '11530', '금천구': '11545', '영등포구': '11560', '동작구': '11590',
        '관악구': '11620', '서초구': '11650', '강남구': '11680', '송파구': '11710', '강동구': '11740'}

## URL request --> 받아오기 ## --> 하루 1000트래픽 한정(1 계정당)
url = 'http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTrade?'
# 서비스키 중요 --> 공공데이터포털에서 오픈API로 받은 인증키를 여기에 입력
serviceKey = 'serviceKey=' + "9rL%2Fv5RbvEC4bLji%2B3hTKF1wuBPkoVk6nnKg54x54CON8OOz5cLFmRFdIWf38wJmwxilBt4XYpuqm0jzpCoM8w%3D%3D" + "&"
LAWD_CD = 'LAWD_CD=' + code['종로구'] + '&' # 법정 코드 번호 --> 가운데 숫자만 변화주면됨. (위 codedict)
DEAL_YMD = 'DEAL_YMD=' + "201801" # 기간 --> 수집시기는 우리의 몫
url_all = url + serviceKey + LAWD_CD + DEAL_YMD
html = req.urlopen(url_all).read().decode('utf-8')
soup = BeautifulSoup(html, 'lxml-xml', from_encoding='utf-8')
price = soup.select_one("item")
print(price)

'''
실거래가 DateFrame을 만드는 게 목적 [소득 추정의 설명력을 높이기 위한 변수 추가]
     아파트  |  동 + 지번  |  거래가  |  전용면적  |  평당 가격  |  층  |
-------------------------------------------------------------------------
      래미안 |  사직동 21  |  78,100  |   29.76   | 78100/29.76 |  4   |

이 Df에도 --> "좌표값 추가"
'''