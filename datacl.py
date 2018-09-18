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
    
    def load_csv(self, csv):
        '''
        csv 파일 불러오기
        '''
        csv=pd.read_csv(csv)
        return csv
    
    def price_clean(self, excel, product):
        '''
        excel 전처리 -> dataframe -> dict // date + 물가
        '''
        alist=self.load_excel(excel).values
        if 'price' not in self.res_dict:
            self.res_dict['price']={i[0]:i[3] for i in alist if product in i[1]}
        else:
            for i in alist:
                if product in i[1]:
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
    
    def oil(self, excel):
        '''
        oil_clean의 3 리스트를 res_dict에 저장
        '''
        gas_list, diesel_list=self.oil_clean(excel)
        for i in gas_list:
            if 'gas' not in self.res_dict:
                self.res_dict['gas']={i[0]: i[1]}
            else:
                self.res_dict['gas'][i[0]]=i[1]
        for i in diesel_list:
            if 'diesel' not in self.res_dict:
                self.res_dict['diesel']={i[0]: i[1]}
            else:
                self.res_dict['diesel'][i[0]]=i[1]
        return
    
    def money(self, excel, category):
        '''
        원달러 환율과 주식을 res_dict에 추가
        '''
        m_list=self.won_clean(excel)
        time_list=self.time_list()
        real_time=list(self.res_dict['price'].keys())
        for i in range(len(m_list)):
            for j in range(len(time_list)):
                if time_list[j] in m_list[i][0]:
                    if category not in self.res_dict:
                        self.res_dict[category]={real_time[j]:m_list[i][1]}
                    else:
                        self.res_dict[category][real_time[j]]=m_list[i][1]
        return
    
    def main(self):
        '''
        편의성을 위한 All-in-one 함수
        '''
        
        self.price_clean("data-set/2018_emart_sq_price.xlsx", "오징어" )
#        self.weather("weather2017.xlsx")
#        self.oil("oil2017.xlsx")
#        self.money("won2017.xlsx", "wondollar")
        self.money("data-set/stock2018.xlsx", "stock")
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
                    if not pd.isna(file.ix[j,2]): tempList.append(file.ix[j,2])
                    rainList.append(file.ix[j,3])
                    if not pd.isna(file.ix[j,4]): windList.append(file.ix[j,4])
#                    print(file.ix[j,1])
#            print(tempList)
            tempRes.append((time, self.mean_list(tempList)))
            rainRes.append((time, self.mean_list(rainList)))
            windRes.append((time, self.mean_list(windList)))
            tempList, rainList, windList=[], [], []
            print (tempRes)
        print("for complete")
        return tempRes, rainRes, windRes
    
    '''
    weather clean
    oil_clean
    '''
    
    def oil_clean(self, excel):
        file=self.load_excel(excel)
        time_list=self.time_list()
        time=list(self.res_dict['price'].keys())
        gas_list, diesel_list=[], []
        for i in range(9, len(file.ix[:,0])):
            file.ix[i,0]=file.ix[i,0].replace("년", "-")
            file.ix[i,0]=file.ix[i,0].replace("월", "-")
            file.ix[i,0]=file.ix[i,0].replace("일", "")
        j=0
        for i in range(9, len(file.ix[:,0])):
            if file.ix[i,0] in time_list:
                gas_list.append((time[j], file.ix[i, 1]))
                diesel_list.append((time[j], file.ix[i, 2]))
                j+=1
        return gas_list, diesel_list
    
    def won_clean(self,excel):
        """
        won 전처리
        """
        won_ex = self.load_excel(excel)
        alist = won_ex.values
        dap = []
        p = re.compile("[\d]{4}-[\d]{2}-[\d]{2}")
        for i in range(len(alist)):
            if "/" in str(alist[i][0]):
                alist[i][0]=str(alist[i][0]).replace("/", "-")
            if p.match(str(alist[i][0]))!=None:
                c=p.match(str(alist[i][0]))
                d =c.string.replace("/","-")
                dap.append((d,alist[i][1]))
        return dap
  
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
        timelist=list(self.get_df(self.res_dict).index)
        pattern=r"[\d]{4}-[\d]{2}-[\d]{2}"
        date_list=[]
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
#    clean=Clean()
#    clean.main()
    clean1=Clean()
    clean1.main()