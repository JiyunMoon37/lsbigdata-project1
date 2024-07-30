import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

economics = pd.read_csv("data/economics.csv")
economics.head()

economics.info()
sns.lineplot(data=economics, x = "date", y="unemploy")
plt.show()

economics["date2"] = pd.to_datetime(economics["date"])
economics.info()

economics[["date", "date2"]]
economics["date2"].dt.year #p111 어트리뷰트 
economics["date2"].dt.month
economics["date2"].dt.day
economics["date2"].dt.month_name() #메서드 
economics["date2"].dt.quarter #분기를 나타낸다. 
economics["quarter"] = economics["date2"].dt.quarter
economics[["date2", "quarter"]]

#각 날짜는 무슨 요일인가?
economics["date2"].dt.day_name()


economics["date2"] + pd.DateOffset(days=30)
economics["date2"] + pd.DateOffset(months=1)


economics["date2"] =  economics["date2"] - pd.Timedelta(days=3)
economics

#p216
#연도 변수 만들기
economics['year'] = economics['date2'].dt.year
economics.head()

#윤년 체크
economics["date2"].dt.is_leap_year 


economics["year"] = economics["date2"].dt.year
sns.lineplot(data=economics, x = 'year', y = 'unemploy')
plt.show()
plt.clf()

economics.head(10)


#scatter
#n = 12, 시그마 / 루트 n 
#각 연도 표준편차, 표본 연도 
import numpy as np
#p217 
sns.lineplot(data = economics, x = 'year' , y = 'unemploy', errorbar = None )
sns.scatterplot(data = economics, x = 'year' , y = 'unemploy', s =2)


my_df = economics.groupby("year", as_index = False) \
        .agg(
            mon_mean = ("unemploy", "mean"),
            mon_std = ("unemploy", "std"),
            mon_n = ("unemploy", "count")
            )
my_df

mean + 1.96*std/sqrt(12) #정규분포에서 95% 신뢰수준에서의 신뢰구간을 설정하기 위한 식 
my_df["left_ci"] = my_df["mon_mean"]- 1.96*my_df["mon_std"]/np.sqrt(my_df["mon_n"])
my_df["right_ci"] = my_df["mon_mean"] + 1.96*my_df["mon_std"]/np.sqrt(my_df["mon_n"])

x = my_df["year"]
y = my_df["mon_mean"]

plt.plot(x, y, color = "black")
plt.show()
plt.clf()

plt.plot(x, y, color = "black")
plt.scatter(x, my_df["left_ci"], color = "blue", s=1)
plt.scatter(x, my_df["right_ci"], color = "red", s=1)
plt.show()
plt.clf()
















