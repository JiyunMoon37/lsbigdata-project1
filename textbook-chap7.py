import pandas as pd 
import numpy as np

df = pd.DataFrame({"sex" : ["M", "F", np.nan, "M", "F"],
                   "score" : [5, 4, 3, 4, np.nan]})
df 

pd.isna(df)

df["score"] + 1
pd.isna(df).sum()

#결측치 제거하기
df.dropna()                         #모든 변수 결측치 제거 
df.dropna(subset = "score")          #score 변수에서 결측치 제거 
df.dropna(subset = ["score", "sex"]) #여러 변수 결측치 제거법

exam = pd.read_csv("data/exam.csv")

#데이터 프레임 location을 사용한 인덱싱 
#exam.loc[행 인덱스, 열 인덱스]
exam.loc[0, 0]
#exam.loc[[0], ["id", "nclass"]]
exam_iloc[0:2, 0:4]
exam.loc[[2, 7, 14], ["math"]] = np.nan
exam.iloc[[2, 7, 4], 2] = np.nan
exam 

df.loc[df["score"]] == 3.0, ["score"]] = 4
df.loc[df["score"] == 3.0, ["score"] = 4]

#수학 점수 50점 이하인 학생들 점수 50점으로 상향 조정!
exam.loc[exam["math"] <= 50, "math"] = 50
exam

#영어 점수 90점 이상 90점으로 하향 조정 (iloc 사용)
#iloc은 조회안됨. 
exam.loc[exam["english"] >= 90, "english"]

#iloc을 사용해서 조회하려면 무조건 숫자 벡터가 들어가야 함. 
exam.iloc[exam["english"] >= 90, 3]              #실행 안됨. 
exam.iloc[np.array(exam["english"] >= 90), 3]    #실행 됨
exam.iloc[np.where(exam["english"] >= 90[0]), 3] #np.where도 튜플이라 [0] 사용해서 꺼내오면 됨. 
exam.iloc[exam[exam["english"] >= 90].index, 3]  #index 벡터도 작동 

#math점수 50점 이하 - 로 변경
exam.loc[exam["math"] <= 50, "math"] = "-"
exam



exam = pd.read_csv("data/exam.csv")
#"-" 결측치를 수학 점수 평균으로 바꾸고 싶은 경우 
#1
math_mean = exam.loc[(exam["math"] != "-"), "math"].mean()
exam.loc[(exam["math"] == "-"), "math"] = math_mean

#2
math_mean = exam.query('math not in ["-"]')['math'].mean()
exam.loc[exam['math']=='-', 'math'] = math_mean

#3
math_mean = exam[(exam["math"] != "-")]["math"].mean()
exam.loc[exam['math']=='-', 'math'] = math_mean

#4
exam.loc[(exam['math'] == "-"), ["math"]] = np.nan 
math_mean = exam["math"].mean()
exam.loc[pd.isna(exam['math'], ['math']] = math_mean 
exam

#5 몰라도 됨. 
vector = np.array([np.nan if x == '-' else float(x) for x in exam["math"]])
vector = np.array([float(x) if x != "-" else np.nan for x in exam["math"]])
exam["math"] = np.where(exam["math"] == "-", math_mean, exam["math"])
exam
 
math_mean = np.nonmean(np.array([np.nan if x == '-' else float(x) for x in exam["math"]])
exam["math"] = np.where(exam["math"] == "-", math_mean, exam["math"])

#6 교재 
math_mean = exam[exam["math"] != "-"]["math"].mean()
exam["math"] = exam["math"].replace("-", math_mean)
exam 


#0722 p193
#!pip install pandas
!pip install seaborn
import pandas as pd 

mpg = pd.read_csv("data/mpg.csv")
mpg.shape

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data = mpg, 
                x = "dipl",
                y = "hwy")
                
plt.show()
plt.clf()

#p201
sns.scatterplot(data = mpg, 
                x = "dipl",
                y = "hwy") \
    .set(xlim=[3,6], ylim = [10, 30])
plt.show()

#수빈이코드
plt.figure(figsize=(5, 4))
sns.scatterplot(data = mpg, 
                x = 'displ', y = 'hwy',
                hue ='drv') \
   .set(xlim =[3, 6], ylim=[10, 30])
plt.show()
        
                
plt.clf()
plt.figure(figsize= (5, 4)) #사이즈 조정 
sns.scatterplot(data = mpg, 
                x = "dipl",
                y = "hwy") \
    .set(xlim=[3,6], ylim = [10, 30])
plt.show()

#막대그래프
df_mpg = mpg.groupby("drv", as_index = False) \
    .agg(mean_hwy = ('hwy', 'mean'))
df_mpg
sns.barplot(data=df_mpg,
            x = "drv", y = "mean_hwy",
            hue = "drv")
plt.show() 


plt.clf()
sns.barplot(data=df_mpg.sort_values("mean_hwy")
            x = "drv", y = "mean_hwy",
            hue = "drv")
plt.show()
#mpg["drv"].unique()

#barplot과 countplot의 차이점 - countplot은 



















               



























