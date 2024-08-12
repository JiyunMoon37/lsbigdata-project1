import pandas as pd 

house_df = pd.read_csv("./data/house/houseprice-with-lonlat.csv")
house_df["Latitude"].mean() #위도 
house_df["Longitude"].mean() #경도 

house_df.info()

#패키지 설치하기 
#!pip install folium

#연습 
import folium

#흰 도화지 맵 가져오기 
map_sig = folium.Map(location = [42.03448223395904, -93.64289689856655],
            zoom_start = 12,
            tiles = "cartodbpositron")

map_sig.save("map_sig.html")


folium.Marker([42.03448223395904,-93.64289689856655], popup = "Ames").add_to(map_sig) 

map_sig.save("map_sig.html")


# 지도 생성: 중심 좌표는 데이터의 평균 위도 및 경도로 설정
map_sig = folium.Map(location=[house_df["Latitude"].mean(), house_df["Longitude"].mean()],
                     zoom_start=12,
                     tiles="cartodbpositron")

#기본코드 
import pandas as pd 
import folium
house_df = pd.read_csv("./data/house/houseprice-with-lonlat.csv")
house_df.columns

house_df[['Latitude', 'Longitude']] #위도, 경도 
#Latitude = house_df["Latitude"]  #위도 
#Longitude = house_df["Longitude"] #경도 

for i in range(len(Latitude)) :
    folium.CircleMarker([Latitude[i],Longitude[i]], radius=3).add_to(map_sig)

map_sig.save("map_sig.html")









