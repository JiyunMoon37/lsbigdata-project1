#0813_team_project
##전체 시각화
#범례 많은 순대로 배치 
import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 연도별 'Exterior_1st' 개수 세기
exterior_distribution = df.groupby(['Year_Built', 'Exterior_1st']).size().unstack(fill_value=0)

# 범례 순서 지정
desired_order = [
    'VinylSd',
    'MetalSd',
    'HdBoard',
    'Wd Sdng',
    'Plywood',
    'CemntBd',
    'BrkFace',
    'WdShing',
    'AsbShng',
    'Stucco',
    'BrkComm',
    'AsphShn',
    'Stone',
    'CBlock',
    'ImStucc',
    'PreCast'
]

# 데이터프레임의 열을 원하는 순서로 재배치
exterior_distribution = exterior_distribution.reindex(columns=desired_order, fill_value=0)

# 시각화
plt.rcParams.update({"font.family": "Malgun Gothic"})
ax = exterior_distribution.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('연도별 외장재 소재 분포')
plt.xlabel('건축 연도')
plt.ylabel('외장재 소재 수')
plt.xticks(rotation=45)

# 범례를 그래프 내부로 이동
plt.legend(title='외장재 소재', loc='upper left', fontsize=10, title_fontsize=12)  # 폰트 크기 조정

plt.tight_layout()
plt.show()
plt.clf()





##그래프 안뜸 

import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 연도별 'Exterior_1st' 개수 세기
exterior_distribution = df.groupby(['Year_Built', 'Exterior_1st']).size().unstack(fill_value=0)

# 지정된 부드러운 색상
colors = {
    "VinylSd": "#FF6F61",  # 부드러운 빨강
    "MetalSd": "#6F9FD8",  # 부드러운 파랑
    "HdBoard": "#F7C94C",  # 부드러운 노랑
    "Wd Sdng": "#6DBE45",  # 부드러운 초록
    "Plywood": "#FFB74D",  # 부드러운 오렌지
    "CemntBd": "#4DB6AC",  # 부드러운 청록
    "BrkFace": "#E57373"   # 부드러운 핑크
}

# 나머지 외장재에 대한 비슷한 톤 색상
additional_colors = {
    "AsbShng": "#FFABAB",  # 부드러운 연한 빨강
    "AsphShn": "#FFDAB9",  # 부드러운 살구색
    "BrkComm": "#D3D3D3",  # 부드러운 회색
    "CBlock": "#A9A9A9",   # 부드러운 다크 그레이
    "PreCast": "#B0E0E6",  # 부드러운 파란색
    "Stone": "#E6E6FA",    # 부드러운 라벤더
    "Stucco": "#FFFACD",   # 부드러운 레몬색
    "WdShing": "#FFCCCB"   # 부드러운 연한 핑크
}

# 색상 매핑
color_map = {exterior: color for exterior, color in colors.items()}
color_map.update({exterior: color for exterior, color in additional_colors.items() if exterior not in colors})

# 시각화
plt.rcParams.update({"font.family": "Malgun Gothic"})
ax = exterior_distribution.plot(kind='bar', stacked=True, figsize=(15, 8), color=[color_map[col] for col in exterior_distribution.columns])

# 제목, 축 레이블 설정
plt.title('연도별 외장재 소재 분포', fontsize=16)
plt.xlabel('건축 연도', fontsize=14)
plt.ylabel('외장재 소재 수', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.show()






##top7 지도 시각화
###이 코드에서 ["VinylSd","MetalSd","HdBoard","Wd Sdng","Plywood","CemntBd","BrkFace"]
###만 색 다양하게 표시하고 나머지는 회색으로 표현해서 다시 시각화해주세요
#빨주노초파남보 설정 
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 연도별 'Exterior_1st' 개수 세기
exterior_distribution = df.groupby(['Year_Built', 'Exterior_1st']).size().unstack(fill_value=0)

# 부드러운 색상으로 지정합니다.
colors = ['#FF6F61', '#6F9FD8', '#F7C94C', '#6DBE45', '#FFB74D', '#4DB6AC', '#E57373']  # 부드러운 색상
highlight_exteriors = ["VinylSd", "MetalSd", "HdBoard", "Wd Sdng", "Plywood", "CemntBd", "BrkFace"]

# 색상 매핑
color_map = {exterior: color for exterior, color in zip(highlight_exteriors, colors)}
color_map.update({exterior: 'lightgray' for exterior in exterior_distribution.columns if exterior not in highlight_exteriors})

# 시각화
plt.rcParams.update({"font.family": "Malgun Gothic"})
ax = exterior_distribution.plot(kind='bar', stacked=True, figsize=(15, 8), color=[color_map[col] for col in exterior_distribution.columns])

# 제목, 축 레이블, 범례의 글자 크기 설정
plt.title('연도별 외장재 소재 분포', fontsize=16)  # 제목 크기
plt.xlabel('건축 연도', fontsize=14)  # x축 레이블 크기
plt.ylabel('외장재 소재 수', fontsize=14)  # y축 레이블 크기
plt.xticks(rotation=45, fontsize=12)  # x축 눈금 크기
plt.yticks(fontsize=12)  # y축 눈금 크기
plt.legend(prop={'size': 10})  # 범례 크기 조정

plt.tight_layout()
plt.show()
plt.clf()






#위에꺼 지도에 시각화~ 나머지는 회색!
import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 외장재 색상 매핑 (주어진 색상으로 업데이트)
exterior_colors = {
    "VinylSd": "#FF6F61",  # 부드러운 빨강
    "MetalSd": "#6F9FD8",  # 부드러운 파랑
    "HdBoard": "#F7C94C",  # 부드러운 노랑
    "Wd Sdng": "#6DBE45",  # 부드러운 초록
    "Plywood": "#FFB74D",  # 부드러운 오렌지
    "CemntBd": "#4DB6AC",  # 부드러운 청록
    "BrkFace": "#E57373"   # 부드러운 핑크
    # 기타 외장재는 기본적으로 회색으로 표시
}

# 지도 초기화
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

Longitude = df['Longitude']
Latitude = df['Latitude']
Price = df['Sale_Price']
Exterior = df['Exterior_1st']

# zip을 사용하여 더 깔끔하게 처리
for lon, lat, price, exterior in zip(Longitude, Latitude, Price, Exterior):
    color = exterior_colors.get(exterior, 'lightgray')  # 기본 색상은 회색
    folium.CircleMarker(
        location=[lat, lon],
        popup=f"Price: ${price}, Exterior: {exterior}",
        radius=3,  # 집의 면적으로 표현
        color=color, 
        fill_color=color,
        fill=True, 
        fill_opacity=0.6
    ).add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house.html')
webbrowser.open_new('map_house.html')






##맨 위에 코드 10년 단위 시각화 
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 연도별 'Exterior_1st' 개수 세기
df['Decade'] = (df['Year_Built'] // 10) * 10  # 연도를 10년 단위로 묶기
exterior_distribution = df.groupby(['Decade', 'Exterior_1st']).size().unstack(fill_value=0)

# 색상을 지정합니다.
colors = ['#FF6F61', '#6F9FD8', '#F7C94C', '#6DBE45', '#FFB74D', '#4DB6AC', '#E57373']  # 부드러운 색상
highlight_exteriors = ["VinylSd", "MetalSd", "HdBoard", "Wd Sdng", "Plywood", "CemntBd", "BrkFace"]

# 색상 매핑
color_map = {exterior: color for exterior, color in zip(highlight_exteriors, colors)}
color_map.update({exterior: 'gray' for exterior in exterior_distribution.columns if exterior not in highlight_exteriors})

# 시각화
plt.rcParams.update({"font.family": "Malgun Gothic"})
ax = exterior_distribution.plot(kind='bar', stacked=True, figsize=(15, 8), color=[color_map[col] for col in exterior_distribution.columns])
plt.title('10년 단위 외장재 소재 분포')
plt.xlabel('건축 연도 (10년 단위)')
plt.ylabel('외장재 소재 수')
plt.xticks(rotation=45)
plt.legend(prop={'size': 8})  # 범례 크기 조정 (8로 설정)
plt.tight_layout()
plt.show()
plt.clf()



#다시 도전
import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 1990년대와 2000년대 데이터 필터링
filtered_df = df[(df['Year_Built'] >= 1990) & (df['Year_Built'] < 2010)]

# 외장재 색상 매핑 (주어진 색상으로 업데이트)
exterior_colors = {
    "VinylSd": "#FF6F61",  # 부드러운 빨강
    "MetalSd": "#6F9FD8",  # 부드러운 파랑
    "HdBoard": "#F7C94C",  # 부드러운 노랑
    "Wd Sdng": "#6DBE45",  # 부드러운 초록
    "Plywood": "#FFB74D",  # 부드러운 오렌지
    "CemntBd": "#4DB6AC",  # 부드러운 청록
    "BrkFace": "#E57373"   # 부드러운 핑크
    # 기타 외장재는 기본적으로 회색으로 표시
}

# 지도 초기화
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

# 필요한 열 추출
Longitude = filtered_df['Longitude']
Latitude = filtered_df['Latitude']
Price = filtered_df['Sale_Price']
Exterior = filtered_df['Exterior_1st']

# zip을 사용하여 더 깔끔하게 처리
for lon, lat, price, exterior in zip(Longitude, Latitude, Price, Exterior):
    color = exterior_colors.get(exterior, 'lightgray')  # 기본 색상은 회색
    folium.CircleMarker(
        location=[lat, lon],
        popup=f"Price: ${price}, Exterior: {exterior}",
        radius=3,  # 집의 면적으로 표현
        color=color, 
        fill_color=color,
        fill=True, 
        fill_opacity=0.6
    ).add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house_1990_2000.html')
webbrowser.open_new('map_house_1990_2000.html')


##1990년대 2000년대-TOP7 이쁜색, 나머지 회색 - 안됨 잠만 
import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 1990년대와 2000년대 데이터 필터링
filtered_df = df[(df['Year_Built'] >= 1990) & (df['Year_Built'] < 2010)]

# 외장재 색상 매핑
exterior_colors = {
    "VinylSd": "#FF6F61",
    "MetalSd": "#6F9FD8",
    "HdBoard": "#F7C94C",
    "Wd Sdng": "#6DBE45",
    "Plywood": "#FFB74D",
    "CemntBd": "#4DB6AC",
    "BrkFace": "#E57373"
}

# 지도 생성
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

# 필요한 열 추출
Longitude = filtered_df['Longitude']
Latitude = filtered_df['Latitude']
Price = filtered_df['Sale_Price']
Exterior = filtered_df['Exterior_1st']

# zip을 사용하여 더 깔끔하게 처리
for lon, lat, price, exterior in zip(Longitude, Latitude, Price, Exterior):
    color = exterior_colors.get(exterior, 'lightgray')  # 기본 색상은 회색
    folium.CircleMarker(
        location=[lat, lon],
        popup=f"Price: ${price}, Exterior: {exterior}",
        radius=3,  # 집의 면적으로 표현
        color=color, 
        fill_color=color,
        fill=True, 
        fill_opacity=0.6
    ).add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house_1990_2000.html')
webbrowser.open_new('map_house_1990_2000.html')












##익스테리어 소재에 따라 연도별로 분류 
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 연도별 'Exterior_1st' 개수 세기
exterior_distribution = df.groupby(['Year_Built', 'Exterior_1st']).size().unstack(fill_value=0)

# 시각화
plt.rcParams.update({"font.family": "Malgun Gothic"})
exterior_distribution.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('연도별 외장재 소재 분포')
plt.xlabel('건축 연도')
plt.ylabel('외장재 소재 수')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()








##10년 단위 자료 개수 
# 10년 단위로 그룹화
df['Decade'] = (df['Year_Built'] // 10) * 10

# 각 10년대별 자료 개수 세기
decade_counts = df['Decade'].value_counts().sort_index()

# 결과 출력
print(decade_counts)




##10년대로 시각화
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 10년 단위로 그룹화
df['Decade'] = (df['Year_Built'] // 10) * 10  # 연도를 10으로 나누고 다시 10을 곱하여 10년 단위로 변환

# 연도별 'Exterior_1st' 개수 세기
exterior_distribution = df.groupby(['Decade', 'Exterior_1st']).size().unstack(fill_value=0)

# 강조할 외장재 소재
highlighted_materials = ['BrkFace', 'CBlock', 'HdBoard', 'MetalSd', 'Plywood', 'VinylSd', 'Wd Sdng']

# 색상 설정: 강조할 소재에 대해 서로 다른 색상을 지정
colors = []
color_map = {
    'BrkFace': 'red',
    'CBlock': 'blue',
    'HdBoard': 'green',
    'MetalSd': 'orange',
    'Plywood': 'purple',
    'VinylSd': 'cyan',
    'Wd Sdng': 'magenta'
}

for material in exterior_distribution.columns:
    if material in highlighted_materials:
        colors.append(color_map[material])  # 강조할 색상
    else:
        colors.append('lightgray')  # 나머지 색상

# 시각화
plt.rcParams.update({"font.family": "Malgun Gothic"})
ax = exterior_distribution.plot(kind='bar', stacked=True, figsize=(15, 8), color=colors)
plt.title('10년 단위 외장재 소재 분포', fontsize=14)  # 제목 글자 크기 조정
plt.xlabel('건축 연도 (10년 단위)', fontsize=12)  # x축 레이블 글자 크기 조정
plt.ylabel('외장재 소재 수', fontsize=12)  # y축 레이블 글자 크기 조정
plt.xticks(rotation=45, fontsize=10)  # x축 눈금 글자 크기 조정
plt.legend(title='외장재 소재', fontsize=10, title_fontsize=12)  # 범례 글자 크기 조정
plt.tight_layout()
plt.show()
plt.clf()







##위도경도랑 엮기-예솔언니 코드 
import pandas as pd
df.columns
df[['Longitude', 'Latitude']].mean()

map_house = folium.Map(location=[42.034482,-93.642897 ],
                    zoom_start=13, tiles='cartodbpositron')
Longitude = df['Longitude']
Latitude = df['Latitude']
Price = df['Sale_Price']

# zip을 쓰면 좀 더 깔끔하게 된다.
for i in range(len(Longitude)):
    folium.CircleMarker([Latitude[i], Longitude[i]],
                        popup=f"Price: ${Price[i]}",
                        radius=3, # 집의 면적으로 표현해보기
                        color='skyblue', 
                        fill_color='skyblue',
                        fill=True, 
                        fill_opacity=0.6 ).add_to(map_house)

map_house.save('map_house.html')
webbrowser.open_new('map_house.html')




# 데이터프레임 df에서 위도, 경도, 주택 가격을 가져옵니다.
Longitude = df['Longitude']
Latitude = df['Latitude']
Price = df['Sale_Price']
Exterior = df['Exterior_1st']

# 10년 단위로 그룹화
df['Decade'] = (df['Year_Built'] // 10) * 10  # 연도를 10으로 나누고 다시 10을 곱하여 10년 단위로 변환

# 외장재 소재 분포를 계산합니다.
exterior_distribution = df.groupby(['Decade', 'Exterior_1st']).size().unstack(fill_value=0)

# 강조할 외장재 소재
highlighted_materials = []

# 색상 설정: 강조할 소재에 대해 서로 다른 색상을 지정
colors = []
color_map = {
    
}

for material in exterior_distribution.columns:
    if material in highlighted_materials:
        colors.append(color_map[material])  # 강조할 색상
    else:
        colors.append('lightgray')  # 나머지 색상

# Folium 지도 생성
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

# 각 주택에 대해 외장재 소재와 가격을 기반으로 원형 마커 추가
for i in range(len(Longitude)):
    # 외장재 소재에 따라 색상 지정
    exterior_material = Exterior[i]
    color = color_map.get(exterior_material, 'lightgray')  # 기본 색상

    folium.CircleMarker(
        location=[Latitude[i], Longitude[i]],
        popup=f"Price: ${Price[i]}<br>Exterior: {exterior_material}",
        radius=3,  # 집의 면적으로 표현해보기
        color=color,
        fill_color=color,
        fill=True,
        fill_opacity=0.6
    ).add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house.html')
webbrowser.open_new('map_house.html')



#모든 연도에 따라 외장재 지도 시각화

import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 외장재 색상 매핑
exterior_colors = {
      # 기타 외장재는 회색으로 표시
}

# 지도 초기화
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

Longitude = df['Longitude']
Latitude = df['Latitude']
Price = df['Sale_Price']
Exterior = df['Exterior_1st']

# zip을 사용하여 더 깔끔하게 처리
for lon, lat, price, exterior in zip(Longitude, Latitude, Price, Exterior):
    color = exterior_colors.get(exterior, 'lightgray')  # 기본 색상은 회색
    folium.CircleMarker(
        location=[lat, lon],
        popup=f"Price: ${price}, Exterior: {exterior}",
        radius=3,  # 집의 면적으로 표현
        color=color, 
        fill_color=color,
        fill=True, 
        fill_opacity=0.6
    ).add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house.html')
webbrowser.open_new('map_house.html')




##1990년대 2000년대
import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 'Year_Built' 열에서 1990년대와 2000년대에 해당하는 데이터 필터링
df['Decade'] = (df['Year_Built'] // 10) * 10
filtered_df = df[(df['Decade'].isin([1990, 2000]))]

# 외장재 색상 매핑
exterior_colors = {
    'BrkFace': 'red',
    'CBlock': 'blue',
    'HdBoard': 'green',
    'MetalSd': 'orange',
    'Plywood': 'purple',
    'VinylSd': 'cyan',
    'Wd Sdng': 'magenta',
    'Other': 'lightgray'  # 기타 외장재는 회색으로 표시
}

# 지도 초기화
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

Longitude = filtered_df['Longitude']
Latitude = filtered_df['Latitude']
Price = filtered_df['Sale_Price']
Exterior = filtered_df['Exterior_1st']

# zip을 사용하여 더 깔끔하게 처리
for lon, lat, price, exterior in zip(Longitude, Latitude, Price, Exterior):
    color = exterior_colors.get(exterior, 'lightgray')  # 기본 색상은 회색
    folium.CircleMarker(
        location=[lat, lon],
        popup=f"Price: ${price}, Exterior: {exterior}",
        radius=3,  # 집의 면적으로 표현
        color=color, 
        fill_color=color,
        fill=True, 
        fill_opacity=0.6
    ).add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house_1990_2000.html')
webbrowser.open_new('map_house_1990_2000.html')

