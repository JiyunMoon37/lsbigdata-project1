#gpt로 코드 돌리기 
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






###이 코드에서 'BrkFace', CBlock','HdBoard', 'MetaISd', 'Plywood', 'VinyISd', 'Wd Sdng' 
###만 색 다양하게 표시하고 나머지는 회색으로 표현해서 다시 시각화해주세요

import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 연도별 'Exterior_1st' 개수 세기
exterior_distribution = df.groupby(['Year_Built', 'Exterior_1st']).size().unstack(fill_value=0)

# 강조할 외장재 소재
highlighted_materials = ['BrkFace', 'HdBoard', 'MetalSd', 'Plywood', 'VinylSd', 'Wd Sdng']

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
exterior_distribution.plot(kind='bar', stacked=True, figsize=(15, 8), color=colors)
plt.title('연도별 외장재 소재 분포', fontsize=12)
plt.xlabel('건축 연도', fontsize=12)
plt.ylabel('외장재 소재 수', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='외장재 소재', fontsize=10, title_fontsize=10)  # 범례 글자 크기 조정
plt.tight_layout()
plt.show()
plt.clf()

print(exterior_distribution)
# 외장재 종류 출력
exterior_types = exterior_distribution.columns.tolist()
print(exterior_types)


##10년 단위 자료 개수 
# 10년 단위로 그룹화
df['Decade'] = (df['Year_Built'] // 10) * 10

# 각 10년대별 자료 개수 세기
decade_counts = df['Decade'].value_counts().sort_index()

# 결과 출력
print(decade_counts)


##0813여기서 민트색 정확한 수치 궁금
# 'VinylSd' 외장재 소재 데이터 필터링
vinyl_sd_data = df[df['Exterior_1st'] == 'VinylSd']

# 10년 단위로 그룹화하여 개수 세기
vinyl_sd_decade_counts = vinyl_sd_data['Decade'].value_counts().sort_index()

# 결과 출력
print(vinyl_sd_decade_counts)

import pandas as pd

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 10년 단위로 그룹화
df['Decade'] = (df['Year_Built'] // 10) * 10

# 각 10년대별 자료 개수 세기
decade_counts = df['Decade'].value_counts().sort_index()

# 결과 출력
print(decade_counts)

# 'VinylSd' 외장재 소재 데이터 필터링
vinyl_sd_data = df[df['Exterior_1st'] == 'VinylSd']

# 10년 단위로 그룹화하여 개수 세기
vinyl_sd_decade_counts = vinyl_sd_data['Decade'].value_counts().sort_index()

# 결과 출력
print(vinyl_sd_decade_counts)








## 'Exterior_2nd' 이거랑 비교하기 
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 연도별 'Exterior_1st' 개수 세기
exterior_distribution = df.groupby(['Year_Built', 'Exterior_2nd']).size().unstack(fill_value=0)

# 시각화
plt.rcParams.update({"font.family": "Malgun Gothic"})
exterior_distribution.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('연도별 외장재 소재 분포', fontsize=12)
plt.xlabel('건축 연도', fontsize=12)
plt.ylabel('외장재 소재 수', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='외장재 소재', fontsize=9, title_fontsize=10)
plt.tight_layout()
plt.show()
plt.clf()









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



##지도+오늘 돌린 코드
import pandas as pd
import folium
import webbrowser
import matplotlib.pyplot as plt

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



##집값에 따라 동그라미 크기 다르게
import pandas as pd
import folium
import webbrowser

# 데이터프레임 df에서 위도, 경도, 주택 가격을 가져옵니다.
Longitude = df['Longitude']
Latitude = df['Latitude']
Price = df['Sale_Price']
Exterior = df['Exterior_1st']

# 10년 단위로 그룹화
df['Decade'] = (df['Year_Built'] // 10) * 10  # 연도를 10으로 나누고 다시 10을 곱하여 10년 단위로 변환

# 색상 설정
highlighted_materials = ['BrkFace', 'CBlock', 'HdBoard', 'MetalSd', 'Plywood', 'VinylSd', 'Wd Sdng']
color_map = {
    'BrkFace': 'red',
    'CBlock': 'blue',
    'HdBoard': 'green',
    'MetalSd': 'orange',
    'Plywood': 'purple',
    'VinylSd': 'cyan',
    'Wd Sdng': 'magenta'
}

# Folium 지도 생성
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

# 각 주택에 대해 외장재 소재와 가격을 기반으로 원형 마커 추가
for i in range(len(Longitude)):
    # 외장재 소재에 따라 색상 지정
    exterior_material = Exterior[i]
    color = color_map.get(exterior_material, 'lightgray')  # 기본 색상

    # 집 값에 따라 동그라미 크기 조정 (예: 10000 달러당 1의 반지름)
    radius = max(2, Price[i] / 10000)  # 최소 반지름을 2로 설정

    folium.CircleMarker(
        location=[Latitude[i], Longitude[i]],
        popup=f"Price: ${Price[i]}<br>Exterior: {exterior_material}",
        radius=radius,  # 집 값에 따라 동그라미 크기 설정
        color=color,
        fill_color=color,
        fill=True,
        fill_opacity=0.5  # 투명도 조정
    ).add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house.html')
webbrowser.open_new('map_house.html')





##0813
#모든 연도에 따라 외장재 지도 시각화

import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

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


##1990년대, 2000년대에 VinylSd 지도 시각화 
import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 'Year_Built' 열에서 1990년대와 2000년대에 해당하는 데이터 필터링
df['Decade'] = (df['Year_Built'] // 10) * 10
filtered_df = df[(df['Decade'].isin([1990, 2000])) & (df['Exterior_1st'] == 'VinylSd')]

# 지도 초기화
map_house = folium.Map(location=[42.034482, -93.642897], zoom_start=13, tiles='cartodbpositron')

Longitude = filtered_df['Longitude']
Latitude = filtered_df['Latitude']
Price = filtered_df['Sale_Price']

# zip을 사용하여 더 깔끔하게 처리
for lon, lat, price in zip(Longitude, Latitude, Price):
    folium.CircleMarker(
        location=[lat, lon],
        popup=f"Price: ${price}, Exterior: VinylSd",
        radius=5,  # 집의 면적으로 표현
        color='cyan', 
        fill_color='cyan',
        fill=True, 
        fill_opacity=0.6
    ).add_to(map_house)

# 지도 저장 및 열기
map_house.save('map_house_vinylsd.html')
webbrowser.open_new('map_house_vinylsd.html')


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



##1950, 1960, 1970년대 
import pandas as pd
import folium
import webbrowser

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 'Year_Built' 열에서 1950년대, 1960년대, 1970년대에 해당하는 데이터 필터링
df['Decade'] = (df['Year_Built'] // 10) * 10
filtered_df = df[df['Decade'].isin([1950, 1960, 1970])]

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
map_house.save('map_house_1950_1970.html')
webbrowser.open_new('map_house_1950_1970.html')























##다른 변수에 의해서 효과 큰 가 알아봄 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 필요한 열만 선택
df_selected = df[['Year_Built', 'Year_Remod_Add', 'Sale_Price', 'Total_Bsmt_SF', 'Gr_Liv_Area']]

# 년도별로 평균값 계산
yearly_data = df_selected.groupby('Year_Built').mean().reset_index()

# 시각화
plt.figure(figsize=(16, 12))

# 각 변수에 대해 서브플롯 생성
plt.subplot(2, 2, 1)
sns.lineplot(data=yearly_data, x='Year_Built', y='Sale_Price', marker='o')
plt.title('Year Built vs Sale Price', fontsize=16)
plt.xlabel('Year Built', fontsize=14)
plt.ylabel('Average Sale Price', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2, 2, 2)
sns.lineplot(data=yearly_data, x='Year_Built', y='Total_Bsmt_SF', marker='o')
plt.title('Year Built vs Total Basement SF', fontsize=16)
plt.xlabel('Year Built', fontsize=14)
plt.ylabel('Average Total Basement SF', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2, 2, 3)
sns.lineplot(data=yearly_data, x='Year_Built', y='Gr_Liv_Area', marker='o')
plt.title('Year Built vs Ground Living Area', fontsize=16)
plt.xlabel('Year Built', fontsize=14)
plt.ylabel('Average Ground Living Area', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

plt.subplot(2, 2, 4)
sns.lineplot(data=yearly_data, x='Year_Built', y='Year_Remod_Add', marker='o')
plt.title('Year Built vs Year Remod/Add', fontsize=16)
plt.xlabel('Year Built', fontsize=14)
plt.ylabel('Average Year Remod/Add', fontsize=14)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)

# 레이아웃 조정
plt.tight_layout(pad=3.0)  # 서브플롯 간의 패딩을 늘림
plt.show()



##건축 연도와 전체 품질 간의 관계 
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 산점도 시각화
plt.figure(figsize=(12, 6))
plt.scatter(df['Year_Built'], df['Overall_Cond'], alpha=0.5)
plt.title('건축 연도와 전체 품질 간의 관계', fontsize=10)  # 제목 글자 크기 조정
plt.xlabel('건축 연도', fontsize=10)  # x축 레이블 글자 크기 조정
plt.ylabel('전체 품질 (Overall Condition)', fontsize=5)  # y축 레이블 글자 크기 조정
plt.xticks(rotation=45, fontsize=9)  # x축 눈금 글자 크기 조정
plt.yticks(fontsize=9)  # y축 눈금 글자 크기 조정
plt.grid()
plt.tight_layout()
plt.show()
plt.clf()


















##익스테리 소재에 따라 언제 리모델링 했는지-이거는 기각  
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')

# 리모델링 연도별 'Exterior_1st' 개수 세기
remod_distribution = df.groupby(['Year_Remod_Add', 'Exterior_1st']).size().unstack(fill_value=0)

# 시각화
plt.rcParams.update({"font.family": "Malgun Gothic"})
remod_distribution.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('리모델링 연도별 외장재 소재 분포')
plt.xlabel('리모델링 연도')
plt.ylabel('외장재 소재 수')
plt.legend(title='Exterior 1st')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


