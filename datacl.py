import pandas as pd
from pandas import Series, DataFrame
import re

# 데이터 전처리 // from datacl import clean
class Clean:
    def __init__(self):
        # 모든 결과물은 self.res_dict에 저장
        self.res_dict={}
        
    def load_excel(self, excel):
        '''
        excel 파일 불러오기
        '''
        excel=pd.read_excel(excel)
        return excel
    
    def price_clean(self, excel, condition):
        '''
        excel 전처리 -> dataframe -> dict // date + 물가 //
        '''
        alist=self.load_excel(excel).values
        if 'price' not in self.res_dict:
            self.res_dict['price']={i[0]:i[3] for i in alist if condition in i[1]}
        else:
            for i in alist:
                if condition in i[1]:
                    self.res_dict['price'][i[0]]=i[3]
        return self.res_dict
        
    def weather(self, excel):
        '''
        temperature excel 전처리 파일(리스트)을 -> date에 맞게 dict추가
        '''
        # time list
        temp_list, rain_list, wind_list=self.weather_clean(excel)
        for i in temp_list:
            if 'temp' not in self.res_dict:
                self.res_dict['temp']={i[0]: i[1]}
            else:
                self.res_dict['temp'][i[0]]=i[1]
        for i in rain_list:
            if 'rain' not in self.res_dict:
                self.res_dict['rain']={i[0]: i[1]}
            else:
                self.res_dict['rain'][i[0]]=i[1]
        for i in wind_list:
            if 'wind' not in self.res_dict:
                self.res_dict['wind']={i[0]: i[1]}
            else:
                self.res_dict['wind'][i[0]]=i[1]
        return self.res_dict
    
    def main(self):
        '''
        편의성을 위한 All-in-one 함수
        '''
        
        self.price_clean("price2017.xlsx", "돼지고기(삼겹살)" )
        self.weather("weather2017.xlsx")
        return self.res_dict
    
    def weather_clean(self, excel):
        '''
        temp 전처리 하여 [날짜: 온도] 처리
        '''      
        # temp 엑셀파일에서 지역과 기온 추출
        file=self.load_excel(excel)
        tempList, rainList, windList=[], [], []
        tempRes, rainRes, windRes=[], [], []
        timelist=list(self.res_dict['price'].keys())
        for time in timelist:
            for j in range(len(file)):
                if time == file.ix[j,1]:
                    tempList.append(file.ix[j,2])
                    rainList.append(file.ix[j,3])
                    windList.append(file.ix[j,4])
                    print(file.ix[j,1])
            print(tempList)
            tempRes.append((time, self.mean_list(tempList)))
            rainRes.append((time, self.mean_list(rainList)))
            windRes.append((time, self.mean_list(windList)))
            tempList, rainList, windList=[], [], []
            print (tempRes)
        print("for complete")
        return tempRes, rainRes, windRes
    
    
    '''
    
    함수 추가 공간
    
    다른 변수//잠복변수 등
    
    '''
    def mean_list(self, m_list):
        '''
        min/max 제거 후 평균 계산 함수
        '''
        m_list.remove(min(m_list))
        m_list.remove(max(m_list))
        mean=sum(m_list)/len(m_list)
        return mean
    
    def time_list(self):
        '''
        정규식을 이용해 timestamp를 time으로 간편화
        '''
        timelist=list(self.res_dict['price'].keys())
        pattern=r"[\d]{4}-[\d]{2}-[\d]{2}"
        date_list=[]
        location_list=[]
        res=[]
        for i in range(len(timelist)):
            date_list.append(re.findall(pattern, str(timelist[i]))[0])
        return date_list
    
    def get_df(self, dictionary):
        '''
        합쳐진 Dict를 DateFrame으로 만들기
        '''
        return DataFrame(dictionary)
    
    def save_excel(self, dictionary, savefile_name="defaultdf.xlsx"):
        '''
        결과물 엑셀로 저장 함수
        '''
        self.get_df(dictionary).to_excel(savefile_name)
        return


if __name__=="__main__":
    clean=Clean()
    clean.main()
    test=clean.load_excel("weather2017.xlsx")