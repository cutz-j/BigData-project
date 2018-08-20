import pandas as pd
from pandas import Series, DataFrame

# 데이터 전처리 // from datacl import clean
class Clean:
    def __init__(self):
        self.res_dict={}
        
    def excel(self, excel):
        excel=pd.read_excel(excel)
        return excel


if __name__=="__main__":
    clean=Clean()
    excel=clean.excel("price2018.xlsx")