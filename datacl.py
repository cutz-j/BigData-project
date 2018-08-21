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
    
    def price_clean(self, excel, category, condition):
        '''
        excel 전처리 -> dataframe -> dict // date + 물가 //
        '''
        alist=self.load_excel(excel).values
        self.res_dict[category]={i[0]: i[3] for i in alist
        if i[1]==condition}
        return self.res_dict
        
    def temp_clean(self, excel, location):
        '''
        temperature excel 전처리 -> date에 맞게 dict추가
        '''
        # time list
        timelist=list(self.res_dict['price'].keys())
        pattern=r"[\d]{4}-[\d]{2}-[\d]{2}"
        for i in range(len(timelist)):
            timelist[i]=re.findall(pattern, str(timelist[i]))[0]
        print(timelist)
        
        file=self.load_excel(excel)
        
        
        
        

if __name__=="__main__":
    clean=Clean()
    res=clean.price_clean("price2018.xlsx", "price", "돼지고기(생삼겹살)")
    res1=clean.load_excel("D:/Github/BigData-project/temp201808.xlsx")
    res2=clean.temp_clean("D:/Github/BigData-project/temp201808.xlsx", "중구")