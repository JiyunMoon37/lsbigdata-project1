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
exterior_distribution.plot(kind='bar', stacked=True, figsize=(15, 8), color=colors)
plt.title('연도별 외장재 소재 분포', fontsize=12)
plt.xlabel('건축 연도', fontsize=12)
plt.ylabel('외장재 소재 수', fontsize=12)
plt.xticks(rotation=45, fontsize=10)
plt.legend(title='외장재 소재', fontsize=10, title_fontsize=10)  # 범례 글자 크기 조정
plt.tight_layout()
plt.show()
plt.clf()

#print(exterior_distribution)





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


