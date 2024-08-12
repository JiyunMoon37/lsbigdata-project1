#0808 4교시 
#데이터 패키지 설치 
#!pip install palmerpenguins
import pandas as pd 
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head() 
penguins["species"].unique()
penguins.columns
#x : bill_length_mm
#y : bill_depth_mm


fig = px.scatter(
    penguins,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "species"
)
fig.show()

#0808 5교시 
#레이아웃 업데이트 
fig.update_layout(
    title = dict(text = "팔머펭귄 종별 부리 길이 vs.깊이"),
    xaxis_title='부리 길이 (mm)',    
    yaxis_title='부리 깊이 (mm)', 
    paper_bgcolor = "black",
    plot_bgcolor = "black",
    font=dict(color="white") 
    )
    
fig.show()

#쌤
# 레이아웃 업데이트
fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", font=dict(color="white")),
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white"),
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white")), 
        tickfont=dict(color="white"),
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(font=dict(color="white")),
)

fig.show()


##레이아웃
#1. 제목크기 키울 것, 
#2. 점 크기 크게,
#3. 범례 제목 "펭귄 종"으로 변경 
fig = px.scatter(
    penguins,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "species"
)
fig.show()

fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", font=dict(color="white", size=24)),  # 제목 폰트 크기 조정
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white", size=14),  # 기본 폰트 크기 조정
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white", size=16)),  # x축 제목 폰트 크기 조정
        tickfont=dict(color="white", size=12),  # x축 눈금 폰트 크기 조정
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white", size=16)),  # y축 제목 폰트 크기 조정
        tickfont=dict(color="white", size=12),  # y축 눈금 폰트 크기 조정
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(font=dict(color="white"), title=dict(text="펭귄 종", font=dict(color="white", size=14)))  # 범례 제목 변경 및 폰트 크기 조정
)

# 점 크기 설정
#fig.update_traces(marker=dict(size=15))  # 점 크기를 15으로 설정

# 점 크기 및 투명도 설정
fig.update_traces(marker=dict(size=10, opacity=0.6))  # 점 크기를 10으로 설정하고 투명도를 0.6으로 설정

fig.show()

#0808 6교시
from sklearn.linear_model import LinearRegression

model = LinearRegression()
penguins = penguins.dropna()
x = penguins[["bill_length_mm"]] #데이터프레임 
y = penguins["bill_depth_mm"]

model.fit(x, y) #데이터 프레임으로 바꾸기 
linear_fit = model.predict(x) 

# #선형회귀 추세선 추가 
# fig.add_trace(
#     go.Scatter(
#         mode = "lines",
#         x = penguins["bill_length_mm"], y = linear_fit,
#         name = "선형회귀직선",
#         line = dict(dash = "dot", color = "white") 
#     )
# )

fig.show()

#각각의 추세선 만들기 
fig = px.scatter(
    penguins,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "species",
    trendline = "ols"
)

fig.update_layout(
    title=dict(text="팔머펭귄 종별 부리 길이 vs. 깊이", font=dict(color="white", size=24)),  # 제목 폰트 크기 조정
    paper_bgcolor="black",
    plot_bgcolor="black",
    font=dict(color="white", size=14),  # 기본 폰트 크기 조정
    xaxis=dict(
        title=dict(text="부리 길이 (mm)", font=dict(color="white", size=16)),  # x축 제목 폰트 크기 조정
        tickfont=dict(color="white", size=12),  # x축 눈금 폰트 크기 조정
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    yaxis=dict(
        title=dict(text="부리 깊이 (mm)", font=dict(color="white", size=16)),  # y축 제목 폰트 크기 조정
        tickfont=dict(color="white", size=12),  # y축 눈금 폰트 크기 조정
        gridcolor='rgba(255, 255, 255, 0.2)'  # 그리드 색깔 조정
    ),
    legend=dict(font=dict(color="white"), title=dict(text="펭귄 종", font=dict(color="white", size=14)))  # 범례 제목 변경 및 폰트 크기 조정
)


# 점 크기 및 투명도 설정
fig.update_traces(marker=dict(size=10, opacity=0.6))  # 점 크기를 10으로 설정하고 투명도를 0.6으로 설정

fig.show()

model.fit(x, y) #데이터 프레임으로 바꾸기 
linear_fit = model.predict(x) 
model.coef_
model.intercept_

#범주형 변수로 회귀분석 진행하기
#범주형 변수인 'species'를 더미 변수로 변환 
penguins_dummies = pd.get_dummies(
    penguins,
    columns = ['species'],
    drop_first = False
    )

penguins_dummies.columns
penguins_dummies.iloc[:, -3:] #새로 생긴 3개 가져옴 

#x와 y 설정
x = penguins_dummies[["bill_length_mm","species_Chinstrap", "species_Gentoo"]]
y = penguins_dummies["bill_depth_mm"]

#모델 학습
model = LinearRegression()
model.fit(x, y)
model.coef_
model.intercept_

#이론적인 회귀직선 식 
y = 0.2 * bill_length - 1.93 * species_Chinstrap - 5.1 * species_Gentoo + 10.56 

#index가 1인 펭귄 부리 깊이 구하는 식 

#x1 : bill_length => 40.5
#x2 : species_Chinstrap => 1 (True)
#x3 : species_Gentoo => 0 (False)
y = 0.2 * 40.5 - 1.93 * True - 5.1 * False + 10.56 

#0808 7교시
regline_y = model.predict(x) 

import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(x["bill_length_mm"], y, color = "black", s=1)
hue = penguins(["species"])
sns.scatterplot(x["bill_length_mm"], regline_y, s=1)

plt.show()
plt.clf()

#gpt 
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 준비
penguins = penguins.dropna()  # 결측치 제거
x = penguins["bill_length_mm"]  # 독립 변수
y = penguins["bill_depth_mm"]    # 종속 변수
hue = penguins["species"]         # 색상 구분을 위한 변수

# 산점도 그리기
sns.scatterplot(x=x, y=y, hue=hue, palette="deep", s=10, color="black")  # 기본 산점도
sns.scatterplot(x=x, y=linear_fit, color="orange", s=1)  # 선형 회귀 직선

plt.show()  # 그래프 표시


















