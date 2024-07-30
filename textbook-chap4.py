import numpy as np
import pandas as pd

df = pd.DataFrame({'name': ["김지훈", "이유진", "박동현", "김민지"],
                'english' : [90, 80, 60, 70],
                'math' : [50, 60, 100,20]
                })

df
df["name"]

type(df)
type(df["name"])

sum(df["english"])



df = pd.DataFrame({'제품' : ["사과", "딸기", "수박"],
                    '가격' : [1800, 1500, 3000],
                    '판매량' : [24, 38, 13]
                    })
                    
df
sum(df["가격"])/3
sum(df["판매량"])/3

!pip install openpyxl
import pandas as pd 
df_exam = pd.read_excel("data/excel_exam.xlsx")
df_exam

sum(df_exam["math"])/20
sum(df_exam["english"])/20
sum(df_exam["science"])/20 

len(df_exam)
df_exam.shape #20행 5열 
df_exam.size

df_exam = pd.read_excel("data/excel_exam.xlsx", 
                        sheet_name = "Sheet2")
df_exam 

df_exam["total"] = df_exam["math"] + df_exam["english"] + df_exam["science"] #pandas series를 집어넣는다. 
df_exam 

df_exam["mean"] = (df_exam["total"]/3).round(1)
df_exam

df_exam[(df_exam["math"] > 50) & (df_exam["english"] > 50)]
(df_exam["math"] > 50) & (df_exam["english"] > 50)

#평균 수학 성적보다 높은데 영어 성적은 평균보다 낮은 사람 
df_exam_mathmean = sum(df_exam"math"])/20
df_exam["math"] > df_exam_mathmean

df_exam_englishmean = sum(df_exam["english"])/20
df_exam["english"] < df_exam_englishmean

df_exam[(df_exam["math"] > df_exam_mathmean) & (df_exam["english"] < df_exam_englishmean)]

#쌤 풀이
mean_m = np.mean(df_exam["math"])
mean_e = np.mean(df_exam["english"])
df_exam[(df_exam["math"] > mean_m) & (df_exam["english"] < mean_e)]

#3반 학생들만 뽑아보기
df_exam[df_exam["nclass"] == 3]
df_nc3 = df_exam[df_exam["nclass"] == 3] #하나의 변수로 정의

df_nc3[["math", "english", "science"]]
df_nc3[1:4]
df_nc3[1:2]

df_exam
df_exam[0:10]
df_exam[7:16] #8번부터 16번까지 뽑고 싶을때 

#홀수번만 뽑고 싶을 때 
df_exam[0:10:2]

#데이터 정렬하고 싶을 때 
df_exam.sort_values("math", ascending = False)
df_exam.sort_values(["nclass", "math"], ascending = [True, False])


a
a > 3
np.where(a >3)
np.where(a >3, "Up", "Down")

df_exam["updown"] = np.where(df_exam["math"] > 50, "Up", "Down")
df_exam 
