import folium
import json
import pandas as pd


# main

# JSON 불러오기
allCenter = json.load(open("d:/project_data/allmap.json", encoding='utf-8'))

# JSON 분리
header = allCenter['DESCRIPTION']
data = allCenter['DATA']

# pd.df로 만들기
alldf = pd.DataFrame(data)
#print(alldf['cot_value_01'])
#print(alldf['cot_coord_x'])
#print(alldf['cot_conts_name'])

# 특정 열만 선택
alldf = alldf[['cot_conts_name', 'cot_coord_x', 'cot_coord_y', 'cot_value_01']]
#print(alldf.dtypes)

# 열 이름 변경
alldf.columns = ['name', 'x', 'y', 'kinds']

# 국공립만 선택
alldf_gov = alldf[alldf['kinds'] == "국공립"]

# folium에 mapping
city_hall = (37.56629, 126.979808)
map_osm = folium.Map(location=city_hall, zoom_start=11)
for i in range(len(alldf_gov)): # 국공립 어린이집
    location = (alldf_gov.iloc[i, 2], alldf_gov.iloc[i, 1]) # 좌표
    folium.Marker(location, popup=alldf_gov.iloc[i, 0]).add_to(map_osm) # marker 이름

# save
map_osm.save("map.html")