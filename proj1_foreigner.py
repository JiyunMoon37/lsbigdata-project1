import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#(단기체류외국인) 월별 단기체류외국인 국적(지역)별 현황


df_foreigner = pd.read_csv('data/foreigner.csv',  encoding='euc-kr')
df_foreigner.columns

#항목 열을 변경(1.변수 이름 변경 했는지?)
df_foreigner = df_foreigner.rename(columns={
    '년': 'year',
    '월': 'month',
    '국적지역': 'nationality',
    '단기체류외국인 수': 'visitors'
})



df_foreigner.describe()

#연도 오름차순 정렬
#d=df_foreigner.sort_values('year')
#d.head(5)

#국적별 빈도수
count=df_foreigner['nationality'].value_counts()
count

#2023년만의 데이터 추출
df_2023=df_foreigner[df_foreigner['year']==2023] 
#->여기서 df_2023을 구할때 그냥 df_foreigner[ ] 하면 boolean 시리즈 형태가 됨. 데이터프레임형태로 만들어주기
#위해 대괄호 를 두번씀.




#연도별 방문자 평균

#d = df_foreigner.groupby('year')['visitors'].mean()
#pd.DataFrame(d)

#2023년 방문자 평균
year_filter=df_foreigner['year']==2023
average_visitors_2023 = df_foreigner.loc[year_filter, 'visitors'].mean()

average_visitors_2023

#2023년 방문자 평균보다 더 많이 방문한 국가 상위 5개
above_average_visitors = df_2023[df_2023['visitors'] > average_visitors_2023]
top_5_countries = above_average_visitors.groupby('nationality')['visitors'].sum().head(5)
top_5_countries
