#p301 

import json
geo = json.load(open('./data/SIG.geojson', encoding = 'UTF-8'))

#행정 구역코드 출력
geo['features'][0]['properties']

#위도, 경도 좌표 출력
geo['features'][0]['geometry']

#p302
import pandas as pd 

df_pop = pd.read_csv('./data/Population_SIG.csv')
df_pop.head()
df_pop.info()

#문자 타입으로 바꿈 
df_pop['code'] = df_pop['code'].astype(str)

#!pip install folium

import folium
folium.Map(location = [35.95, 127.7],
           zoom_start = 8)

#0807 1교시 
#p308
import json 
geo_seoul = json.load(open('./data/SIG_Seoul.geojson', encoding = 'UTF-8'))

type(geo_seoul)
len(geo_seoul)

geo_seoul.keys() #'type', 'name', 'crs', 'features' 
geo_seoul['type'] 
geo_seoul['name']
geo_seoul['crs']
geo_seoul['features']

geo_seoul['features'][0]
len(geo_seoul['features'][0])

geo_seoul['features'][0].keys()

#행정 구역 코드 출력 
geo_seoul['features'][0]['properties']

#위도, 경도 좌표 출력
geo_seoul['features'][0]['geometry']

type(geo_seoul['features'][0]['geometry']['coordinates'])


#안에 숫자 바꾸면 구를 바꿀 수 있음. 
coordinate_list = geo_seoul['features'][4]['geometry']['coordinates']

#len(coordinate_list)
#len(coordinate_list[0])
#len(coordinate_list[0][0])

import numpy as np
coordinate_array = np.array(coordinate_list[0][0])
#coordinate_array

x = coordinate_array[:,0] #행 전체, 첫 번째 열 한개 
y = coordinate_array[:,1] #행 전체, 두 번째 열 한개 

import matplotlib.pyplot as plt
plt.plot(x[::10], y[::10])
plt.show()
plt.clf() 

#함수로 만들기
def draw_seoul(num) :
    gu_name = geo_seoul['features'][num]['properties']["SIG_KOR_NM"]
    coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_Array = np.array(coordinate_list[0][0])
    x = coordinate_array[:,0] 
    y = coordinate_array[:,1]
    
    plt.rcParams.update({"font.family":"Malgun Gothic"})
    plt.plot(x, y)
    plt.title(gu_name) 
    plt.show()
    plt.clf()
    
draw_seoul(18) 


#0807 3교시 
#서울시 전체 지도 그리기
# gu_name
# plt.plot(x, y, hue = "gu_name") 


#구 개수 구하는 코드 
gu_count = len(geo_seoul['features'])
gu_count

#서울시 전체 지도 그리기 함수

def draw_seoul_map():
    plt.figure(figsize=(12, 12))  # 그림 크기 설정
    
    for feature in geo_seoul['features']:
        gu_name = feature['properties']["SIG_KOR_NM"]
        coordinate_list = feature["geometry"]["coordinates"]

        # 다각형의 좌표가 여러 개일 수 있으므로 반복
        for coords in coordinate_list:
            coordinate_array = np.array(coords[0])  # 첫 번째 다각형의 좌표
            x = coordinate_array[:, 0]
            y = coordinate_array[:, 1]

            plt.plot(x, y)  # 각 구를 플롯에 추가
            
    plt.rcParams.update({"font.family": "Malgun Gothic"})
    plt.title("서울시 전체 지도")
    plt.xlabel("경도")
    plt.ylabel("위도")
    plt.grid()
    plt.show()

# 함수 호출
draw_seoul_map()

#0807 4교시
#gu_name = geo_seoul["features"][0]["properties"]["SIG_KOR_NM"]
#방법 1 
gu_name = list()

for i in range(25) :
    gu_name.append(geo_seoul["features"][i]["properties"]["SIG_KOR_NM"])

gu_name 

#방법 2 
gu_name = [geo_seoul["features"][i]["properties"]["SIG_KOR_NM"] for i in range(25))]
gu_name 

#x, y 판다스 데이터 프레임
import pandas as pd

#방법 3
def make_seouldf(num):
    gu_name = geo_seoul["features"][num]["properties"]["SIG_KOR_NM"]
    coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
    coordinate_array = np.array(coordinate_list[0][0])
    x = coordinate_array[:,0]
    y = coordinate_array[:,1]
    
    return pd.DataFrame({"gu_name" :gu_name, "x" : x, "y" : y})

make_seouldf(1)

result = pd.DataFrame({}) 
for i in range(25):
    result = pd.concat([result, make_seouldf[i], ignore_index = True)

result

#df_a = pd.concat([df_a, df_b])

#데이터프레임 concat 예제 
df_a = pd.DataFrame({
    'gu_name' : [],
    'x' : [],
    'y' : []
})

#0807 5교시
result = pd.DataFrame({}) 
for i in range(25):
    result = pd.concat([result, make_seouldf(i)], ignore_index = True)

result

#scatter로 표현 
import seaborn as sns
sns.scatterplot(
    data = result,
    x = 'x', y = 'y', hue = 'gu_name', legend = False,
    palette = "deep", s=2
    )
 
plt.show()
plt.clf() 

#서울 그래프 그리기
import seaborn as sns
gangnam_df = result.assign(is_gangnam = np.where(result["gu_name"]=="강남구", "강남", "안강남"))

sns.scatterplot(
    data = gangnam_df,
    x = 'x', y = 'y', legend = False, 
    palette = {"강남": "red", "안강남": "grey"},
    hue = 'is_gangnam', s=2,
    )
plt.show()
plt.clf()

gangnam_df["is_gangnam"].unique()


 #주영
gu_name = []
for i in range(len(geo_seoul["features"])):
  gu_name.append(geo_seoul["features"][i]['properties']["SIG_KOR_NM"])

def draw_seoul(num):
  gu_name = geo_seoul["features"][num]['properties']["SIG_KOR_NM"]
  coordinate_list = geo_seoul["features"][num]["geometry"]["coordinates"]
  coordinate_array = np.array(coordinate_list[0][0])
  x = coordinate_array[:,0]
  y = coordinate_array[:,1]
  
  return pd.DataFrame({"gu_name": gu_name, "x":x, "y":y})
draw_seoul(12)

result = pd.DataFrame({
  "gu_name": [],
  "x":[],
  "y":[]
}) 
for i in range(25):
  result = pd.concat([result,draw_seoul(i)], ignore_index = True)
result

#0807 6교시
geo_seoul = json.load(open('./data/SIG_Seoul.geojson', encoding = 'UTF-8'))
#행정 구역 코드 출력 
geo_seoul['features'][0]['properties']

df_pop = pd.read_csv("data/Population_SIG.csv")
df_pop.head() 
df_seoulpop = df_pop.iloc[1:26]
df_seoulpop["code"] = df_seoulpop["code"].astype(str)
df_seoulpop.info()

#패키지 설치하기 
#!pip install folium
#p303 
import folium

center_x = result["x"].mean()
center_y = result["y"].mean()
map_sig = folium.Map(location = [37.551, 126.973],
            zoom_start = 12,
            tiles = "cartodbpositron")

map_sig.save("map_sig.html")

#코로플릿
folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ("code", "pop"),
    key_on = "feature.properties.SIG_CD").add_to(map_sig)

map_sig.save("map_sig.html")

#0807 7교시
geo_seoul
geo_seoul["features"][0]["properties"]["SIG_CD"]


#더 정확하게 코로플릿 표시 
folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ("code", "pop"),
    bins = bins, 
    key_on = "feature.properties.SIG_CD").add_to(map_sig)

map_sig.save("map_sig.html")

#p306 
#인구수를 기준에 따라 표시하기 (5단계)
bins = list(df_seoulpop["pop"].quantile([0, 0.2, 0.4, 0.6, 0.8, 1]))

folium.Choropleth(
    geo_data = geo_seoul,
    data = df_seoulpop,
    columns = ("code", "pop"),
    fill_color = "viridis", 
    bins = bins, 
    key_on = "feature.properties.SIG_CD").add_to(map_sig)

map_sig.save("map_sig.html")

#점 찍는 법 
make_seouldf(0) #종로구 
make_seouldf(0).iloc[:,1:3].mean() #x : 126.983800, y :37.583744
folium.Marker([37.583744,126.983800], popup = "종로구").add_to(map_sig) 

map_sig.save("map_sig.html")











