#0812 2교시 
#데이터 패키지 설치 
#!pip install palmerpenguins
import pandas as pd 
import numpy as np
import plotly.express as px
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head() 

fig = px.scatter(
    penguins,
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "species"
)
fig.show()

fig.update_layout(
    title = {'text' : "<span style = 'font-weight:bold'> ... </span> 팔머펭귄" </span>
             'x' : 0.5}
)

#css 문법 이해하기 
#<span> ... </span> #글자의 서식을 나타내는 구문 
#<span style = 'font-weight:bold'> ... </span> #글자의 서식을 나타내는 구문 
#<span style='color:blue','font-weight:bold'> 
