#필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

##필요한 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')
df.columns #행 뭐있나 알아보기


#지붕 스타일
df['Roof_Style']

# Neighborhood 별 Roof_Style 개수 세기
roof_distribution = df.groupby(['Neighborhood', 'Roof_Style']).size().unstack(fill_value=0)

# 한글 폰트 설정
plt.rcParams.update({"font.family": "Malgun Gothic"})

# 시각화
roof_distribution.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('지역별 지붕 스타일 분포')
plt.xlabel('Neighborhood')
plt.ylabel('지붕 스타일 수')
plt.legend(title='Roof Style')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf()


#연도별 지붕 스타일
df['YearBuilt']

# 연도별 Roof_Style 개수 세기
roof_distribution_year = df.groupby(['Year_Built', 'Roof_Style']).size().unstack(fill_value=0)

# 시각화
roof_distribution_year.plot(kind='bar', stacked=True, figsize=(15, 8))
plt.title('연도별 지붕 스타일 분포')
plt.xlabel('연도')
plt.ylabel('지붕 스타일 수')
plt.legend(title='Roof Style')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
plt.clf() 


#품질과 리모델링 연도 관련성 - 그래프 잘 안나옴. 
#'Overall_Cond''Year_Remod_Add''Roof_Matl'
import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
df = pd.read_csv('./data/house/houseprice-with-lonlat.csv')
df["Year_Remod_Add"]
# YearRemodAdd 별 OverallQual 평균 계산
qual_distribution = df.groupby('Year_Remod_Add')['Roof_Matl'].mean()

# 시각화
qual_distribution.plot(kind='bar', figsize=(15, 8), color='skyblue')
plt.title('연도별 Overall Quality 평균')
plt.xlabel('리모델링 연도')
plt.ylabel('Overall Quality 평균')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()





