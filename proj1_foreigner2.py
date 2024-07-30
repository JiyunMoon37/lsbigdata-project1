import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#(단기체류외국인) 월별 단기체류외국인 국적(지역)별 현황

df_foreigner = pd.read_csv('data/foreigner.csv',  encoding='euc-kr')
df_foreigner.columns

#1.변수 이름 변경 했는지?
df_foreigner = df_foreigner.rename(columns={
    '년': 'year',
    '월': 'month',
    '국적지역': 'nationality',
    '단기체류외국인 수': 'visitors'
})

df_foreigner.head()

#2. 행들을 필터링 했는지?
df_foreigner.query("year == 2023").head()

#3. 새로운 변수를 생성했는지?
new = df_foreigner.query("year == 2023").assign(
    average_visitors_2023 = df_foreigner['visitors'].sum() / df_foreigner['visitors'])
new

#4. 그룹 변수 기준으로 요약을 했는지?)
new2 = new.query('visitors > average_visitors_2023')
new2

#5. 정렬했는지?
new2.sort_values('visitors', ascending = False)


