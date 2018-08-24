import pandas as pd
from pandas import Series, DataFrame
import re

# 데이터 전처리 // from datacl import clean
class Clean:
    def __init__(self):
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
        
    def weather(self, excel, category, location="남산"):
        '''
        temperature excel 전처리 파일(리스트)을 -> date에 맞게 dict추가
        '''
        # time list
        temp_list=self.weather_clean(excel, location)
        for i in temp_list:
            if category not in self.res_dict:
                self.res_dict[category]={i[0]: i[1]}
            else:
                self.res_dict[category][i[0]]=i[1]
        return self.res_dict
    
    def main(self):
        '''
        편의성을 위한 All-in-one 함수
        '''
        for cat in ['price','temp','rain','hum','wsd']:
            for year in range(2015,2019):
                if cat == 'price':
                    self.price_clean("D:/Github/BigData-project/data-set/"+cat+str(year)+".xlsx", "돼지고기(삼겹살)" )
                else: 
                    for month in range(1,13):
                        if year == 2018:
                            if month > 8:
                                break
                        self.weather("D:/Github/BigData-project/data-set/"+cat+str(year)+str(month)+".xlsx",cat, "은평")
        return self.res_dict
    
    def weather_clean(self, excel, location="남산"):
        '''
        temp 전처리 하여 [날짜: 온도] 처리
        '''
        # 정규식으로 물가Dict의 날짜 추출
        timelist=list(self.res_dict['price'].keys())
        pattern=r"[\d]{4}-[\d]{2}-[\d]{2}"
        date_list=[]
        location_list=[]
        res=[]
        for i in range(len(timelist)):
            date_list.append(re.findall(pattern, str(timelist[i]))[0])
        # temp 엑셀파일에서 지역과 기온 추출
        file=self.load_excel(excel)
        for i in file.values[1:]:
            if "[서] " +location in i[0]:
                location_list=i
        if len(location_list)==0:
            print("잘못된 지역명입니다.")
        # temp를 (날짜, 기온)으로 리턴 (결측치 제거)
        date=pd.Categorical(self.load_excel(excel))[0][-8:-1]
        for j in range(len(location_list)):
            if j == 0:
                continue
            elif j < 10:
                date_res=date+"-0"+str(j)
            else:
                date_res=date+"-"+str(j)
            if date_res in date_list:
                if not location_list[j]=='-':
                    res.append((timelist[date_list.index(date_res)], location_list[j]))
        return res
    
    '''
    
    함수 추가 공간
    
    다른 변수//잠복변수 등
    
    '''
    
    def get_df(self, dictionary):
        '''
        합쳐진 Dict를 DateFrame으로 만들기
        '''
        return DataFrame(dictionary)
    
    def save_excel(self, dictionary, savefile_name="defaultdf.xlsx"):
        self.get_df(dictionary).to_excel(savefile_name)
        return


if __name__=="__main__":
    clean=Clean()
    clean.main()
    res=clean.res_dict
    df=clean.get_df(res)