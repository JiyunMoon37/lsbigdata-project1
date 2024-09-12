#0814_team_project
#연도별 외장재 소재 분포
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
ax = exterior_distribution.plot(kind='bar', stacked=True, figsize=(9,7))  # 폭 조정
plt.title('연도별 외장재 소재 분포', fontsize=10)
plt.xlabel('건축 연도', fontsize=3)
plt.ylabel('외장재 소재 수', fontsize=8)

# 연도를 세로로 표시 및 크기 조정
plt.xticks(rotation=90, fontsize=3)

# 범례를 그래프 오른쪽에 배치 및 크기 조정
plt.legend(title='외장재 소재', loc='upper left', fontsize=5, title_fontsize=8, bbox_to_anchor=(1, 1))

# 여백 조정
plt.subplots_adjust(right=1)  # 오른쪽 여백을 더 늘림

plt.tight_layout()
plt.show()
plt.clf()

#plotly.express
import pandas as pd
import plotly.express as px

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 연도별 'Exterior_1st' 개수 세기
exterior_distribution = df.groupby(['Year_Built', 'Exterior_1st']).size().reset_index(name='Count')

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

# 외장재 소재의 순서를 재정렬
exterior_distribution['Exterior_1st'] = pd.Categorical(exterior_distribution['Exterior_1st'], categories=desired_order, ordered=True)
exterior_distribution = exterior_distribution.sort_values('Exterior_1st')

# 시각화
fig = px.bar(exterior_distribution, 
             x='Year_Built', 
             y='Count', 
             color='Exterior_1st', 
             title='연도별 외장재 소재 분포',
             labels={'Count': '외장재 소재 수', 'Year_Built': '건축 연도'},
             text='Count')

# 그래프 설정
fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(barmode='stack', 
                  xaxis_title='건축 연도', 
                  yaxis_title='외장재 소재 수', 
                  legend_title='외장재 소재',
                  font=dict(family='Malgun Gothic', size=10))

# 그래프 보여주기
fig.show()






#연도별 전체 TOP7 
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 10년 단위로 연도 그룹화
df['Decade'] = (df['Year_Built'] // 10) * 10  # 10년 단위로 변환
exterior_distribution = df.groupby(['Decade', 'Exterior_1st']).size().unstack(fill_value=0)

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

# 색상 지정
colors = [
    '#FF6F61',  # VinylSd
    '#6F9FD8',  # MetalSd
    '#F7C94C',  # HdBoard
    '#6DBE45',  # Wd Sdng
    '#FFB74D',  # Plywood
    '#4DB6AC',  # CemntBd
    '#9B59B6'   # BrkFace
] + ['#D3D3D3'] * (len(exterior_distribution.columns) - 7)  # 나머지 색상은 회색으로 설정

# 시각화
plt.rcParams.update({"font.family": "Malgun Gothic"})
ax = exterior_distribution.plot(kind='bar', stacked=True, figsize=(9, 7), color=colors)  # 색상 적용
plt.title('10년 단위 연도별 외장재 소재 분포', fontsize=10)
plt.xlabel('건축 10년대', fontsize=3)
plt.ylabel('외장재 소재 수', fontsize=8)

# 연도를 세로로 표시 및 크기 조정
plt.xticks(rotation=0, fontsize=6)

# 범례를 그래프 오른쪽에 배치 및 크기 조정
plt.legend(title='외장재 소재', loc='upper right', fontsize=5, title_fontsize=8)

# 여백 조정
plt.subplots_adjust(right=0.8)  # 오른쪽 여백 조정

plt.tight_layout()
plt.show()
plt.clf()


#plotly
import pandas as pd
import plotly.express as px

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 연도별 'Exterior_1st' 개수 세기
exterior_distribution = df.groupby(['Year_Built', 'Exterior_1st']).size().reset_index(name='Count')

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

# 외장재 소재의 순서를 재정렬
exterior_distribution['Exterior_1st'] = pd.Categorical(exterior_distribution['Exterior_1st'], categories=desired_order, ordered=True)
exterior_distribution = exterior_distribution.sort_values('Exterior_1st')

# 색상 지정
colors = [
    '#FF6F61',  # VinylSd
    '#6F9FD8',  # MetalSd
    '#F7C94C',  # HdBoard
    '#6DBE45',  # Wd Sdng
    '#FF8C00',  # Plywood
    '#4DB6AC',  # CemntBd
    '#9B59B6',  # BrkFace
] + ['#D3D3D3'] * (len(exterior_distribution['Exterior_1st'].unique()) - 7)  # 나머지 색상은 회색으로 설정

# 시각화
fig = px.bar(exterior_distribution, 
             x='Year_Built', 
             y='Count', 
             color='Exterior_1st', 
             title='연도별 외장재 소재 분포',
             labels={'Count': '외장재 소재 수', 'Year_Built': '건축 연도'},
             text='Count',
             color_discrete_sequence=colors)

# 그래프 설정
fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(barmode='stack', 
                  xaxis_title='건축 연도', 
                  yaxis_title='외장재 소재 수', 
                  legend_title='외장재 소재',
                  font=dict(family='Malgun Gothic', size=10))

# 그래프 보여주기
fig.show()



#plotly_10년 단위 
import pandas as pd
import plotly.express as px

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 10년 단위로 연도 그룹화
df['Decade'] = (df['Year_Built'] // 10) * 10  # 10년 단위로 변환
exterior_distribution = df.groupby(['Decade', 'Exterior_1st']).size().reset_index(name='Count')

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

# 외장재 소재의 순서를 재정렬
exterior_distribution['Exterior_1st'] = pd.Categorical(exterior_distribution['Exterior_1st'], categories=desired_order, ordered=True)
exterior_distribution = exterior_distribution.sort_values('Exterior_1st')

# 색상 지정
colors = [
    '#FF6F61',  # VinylSd
    '#6F9FD8',  # MetalSd
    '#F7C94C',  # HdBoard
    '#6DBE45',  # Wd Sdng
    '#FF8C00',  # Plywood
    '#4DB6AC',  # CemntBd
    '#9B59B6',  # BrkFace
] + ['#D3D3D3'] * (len(exterior_distribution['Exterior_1st'].unique()) - 7)  # 나머지 색상은 회색으로 설정

# 시각화
fig = px.bar(exterior_distribution, 
             x='Decade', 
             y='Count', 
             color='Exterior_1st', 
             title='10년 단위 연도별 외장재 소재 분포',
             labels={'Count': '외장재 소재 수', 'Decade': '건축 10년대'},
             text='Count',
             color_discrete_sequence=colors)

# 그래프 설정
fig.update_traces(texttemplate='%{text}', textposition='outside')
fig.update_layout(barmode='stack', 
                  xaxis_title='건축 10년대', 
                  yaxis_title='외장재 소재 수', 
                  legend_title='외장재 소재',
                  font=dict(family='Malgun Gothic', size=10))

# 그래프 보여주기
fig.show()




#Ames 지역 주택 외장재 분포 
import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 외장재 색상 매핑
exterior_colors = {
    'VinylSd': '#FF6F61',
    'MetalSd': '#6F9FD8',
    'HdBoard': '#F7C94C',
    'Wd Sdng': '#6DBE45',
    'Plywood': '#FF8C00',
    'CemntBd': '#4DB6AC',
    'BrkFace': '#9B59B6',
    'WdShing': '#D3D3D3',
    'AsbShng': '#D3D3D3',
    'Stucco': '#D3D3D3',
    'BrkComm': '#D3D3D3',
    'AsphShn': '#D3D3D3',
    'Stone': '#D3D3D3',
    'CBlock': '#D3D3D3',
    'ImStucc': '#D3D3D3',
    'PreCast': '#D3D3D3'
}

# 지도 생성
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

# 집의 좌표 및 가격, 외장재 정보를 zip을 사용하여 더 깔끔하게 처리
Longitude = df['Longitude']
Latitude = df['Latitude']
Price = df['Sale_Price']
Exterior = df['Exterior_1st']

for lon, lat, price, exterior in zip(Longitude, Latitude, Price, Exterior):
    color = exterior_colors.get(exterior, 'lightgray')  # 외장재에 해당하는 색상, 기본 색상은 회색
    folium.CircleMarker(
        location=[lat, lon],
        popup=f"Price: ${price:,.2f}, Exterior: {exterior}",
        radius=5,  # 집의 면적으로 표현
        color=color, 
        fill_color=color,
        fill=True, 
        fill_opacity=0.6
    ).add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house.html')
webbrowser.open_new('map_house.html')


#그 소재 누르면 그색만 나오게 지도 시각화
import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 외장재 색상 매핑
exterior_colors = {
    'VinylSd': '#FF6F61',
    'MetalSd': '#6F9FD8',
    'HdBoard': '#F7C94C',
    'Wd Sdng': '#6DBE45',
    'Plywood': '#FF8C00',
    'CemntBd': '#4DB6AC',
    'BrkFace': '#9B59B6',
    'WdShing': '#D3D3D3',
    'AsbShng': '#D3D3D3',
    'Stucco': '#D3D3D3',
    'BrkComm': '#D3D3D3',
    'AsphShn': '#D3D3D3',
    'Stone': '#D3D3D3',
    'CBlock': '#D3D3D3',
    'ImStucc': '#D3D3D3',
    'PreCast': '#D3D3D3'
}

# 지도 생성
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

# 집의 좌표 및 가격, 외장재 정보를 zip을 사용하여 더 깔끔하게 처리
Longitude = df['Longitude']
Latitude = df['Latitude']
Price = df['Sale_Price']
Exterior = df['Exterior_1st']

# 외장재별로 레이어를 생성
layers = {}
for exterior in exterior_colors.keys():
    layers[exterior] = folium.FeatureGroup(name=exterior)

    for lon, lat, price, ext in zip(Longitude, Latitude, Price, Exterior):
        if ext == exterior:
            color = exterior_colors.get(ext, 'lightgray')  # 외장재에 해당하는 색상
            layers[exterior].add_child(
                folium.CircleMarker(
                    location=[lat, lon],
                    popup=f"Price: ${price:,.2f}, Exterior: {ext}",
                    radius=5,  # 집의 면적으로 표현
                    color=color, 
                    fill_color=color,
                    fill=True, 
                    fill_opacity=0.6
                )
            )

# 각 레이어를 지도에 추가
for layer in layers.values():
    layer.add_to(map_house)

# 레이어 컨트롤 추가
folium.LayerControl().add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house.html')
webbrowser.open_new('map_house.html')




#1990년대 필터링
import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 1990년대 집 필터링 (여기서는 Year_Built 열이 있다고 가정)
df_1990s = df[(df['Year_Built'] >= 1990) & (df['Year_Built'] < 2000)]

# 외장재 색상 매핑
exterior_colors = {
    'VinylSd': '#FF6F61',
    'MetalSd': '#6F9FD8',
    'HdBoard': '#F7C94C',
    'Wd Sdng': '#6DBE45',
    'Plywood': '#FF8C00',
    'CemntBd': '#4DB6AC',
    'BrkFace': '#9B59B6',
    'WdShing': '#D3D3D3',
    'AsbShng': '#D3D3D3',
    'Stucco': '#D3D3D3',
    'BrkComm': '#D3D3D3',
    'AsphShn': '#D3D3D3',
    'Stone': '#D3D3D3',
    'CBlock': '#D3D3D3',
    'ImStucc': '#D3D3D3',
    'PreCast': '#D3D3D3'
}

# 지도 생성
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

# 집의 좌표 및 가격, 외장재 정보를 zip을 사용하여 더 깔끔하게 처리
Longitude = df_1990s['Longitude']
Latitude = df_1990s['Latitude']
Price = df_1990s['Sale_Price']
Exterior = df_1990s['Exterior_1st']

# 외장재별로 레이어를 생성
layers = {}
for exterior in exterior_colors.keys():
    layers[exterior] = folium.FeatureGroup(name=exterior)

    for lon, lat, price, ext in zip(Longitude, Latitude, Price, Exterior):
        if ext == exterior:
            color = exterior_colors.get(ext, 'lightgray')  # 외장재에 해당하는 색상
            layers[exterior].add_child(
                folium.CircleMarker(
                    location=[lat, lon],
                    popup=f"Price: ${price:,.2f}, Exterior: {ext}",
                    radius=5,  # 집의 면적으로 표현
                    color=color, 
                    fill_color=color,
                    fill=True, 
                    fill_opacity=0.6
                )
            )

# 각 레이어를 지도에 추가
for layer in layers.values():
    layer.add_to(map_house)

# 레이어 컨트롤 추가
folium.LayerControl().add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house.html')
webbrowser.open_new('map_house.html')




#2000년대만 시각화
import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 2000년대 집 필터링 (여기서는 Year_Built 열이 있다고 가정)
df_2000s = df[(df['Year_Built'] >= 2000) & (df['Year_Built'] < 2010)]

# 외장재 색상 매핑
exterior_colors = {
    'VinylSd': '#FF6F61',
    'MetalSd': '#6F9FD8',
    'HdBoard': '#F7C94C',
    'Wd Sdng': '#6DBE45',
    'Plywood': '#FF8C00',
    'CemntBd': '#4DB6AC',
    'BrkFace': '#9B59B6',
    'WdShing': '#D3D3D3',
    'AsbShng': '#D3D3D3',
    'Stucco': '#D3D3D3',
    'BrkComm': '#D3D3D3',
    'AsphShn': '#D3D3D3',
    'Stone': '#D3D3D3',
    'CBlock': '#D3D3D3',
    'ImStucc': '#D3D3D3',
    'PreCast': '#D3D3D3'
}

# 지도 생성
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

# 집의 좌표 및 가격, 외장재 정보를 zip을 사용하여 더 깔끔하게 처리
Longitude = df_2000s['Longitude']
Latitude = df_2000s['Latitude']
Price = df_2000s['Sale_Price']
Exterior = df_2000s['Exterior_1st']

# 외장재별로 레이어를 생성
layers = {}
for exterior in exterior_colors.keys():
    layers[exterior] = folium.FeatureGroup(name=exterior)

    for lon, lat, price, ext in zip(Longitude, Latitude, Price, Exterior):
        if ext == exterior:
            color = exterior_colors.get(ext, 'lightgray')  # 외장재에 해당하는 색상
            layers[exterior].add_child(
                folium.CircleMarker(
                    location=[lat, lon],
                    popup=f"Price: ${price:,.2f}, Exterior: {ext}",
                    radius=5,  # 집의 면적으로 표현
                    color=color, 
                    fill_color=color,
                    fill=True, 
                    fill_opacity=0.6
                )
            )

# 각 레이어를 지도에 추가
for layer in layers.values():
    layer.add_to(map_house)

# 레이어 컨트롤 추가
folium.LayerControl().add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house.html')
webbrowser.open_new('map_house.html')




