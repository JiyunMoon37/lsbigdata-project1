import pandas as pd 
import numpy as np

#데이터 탐색 함수
#head()
#tail()
#shape
#info()
#describe()

exam = pd.read_csv("data/exam.csv")
exam.head(10)
exam.tail(10)
exam.shape #shape은 어트리뷰트라 ()쓰면 안된다. 
exam.shape
exam.info()
exam.describe()

type(exam)
var = [1,2,3]
type
exam
var

#매서드 vs 속성(어트리뷰트)
#매서드는 코드를 쳤을 때 ()가 안떠서 함수 임을 알 수 있지만
#속성은 코드를 쳤을 때 ()가 안뜨는 속성 자체임. 

exam2 = exam.copy()
exam2 = exam2.rename(columns={"nclass" : "class"})
exam2

exam2["total"] = exam2["math"] + exam2["english"] + exam2["science"]
exam2.head()

#200이상이면 pass, 200미만이면 fail 
exam2["test"] = np.where(exam2["total"] >= 200, "pass", "fail")
exam2.head()

exam2["test"] #series로 접근 
exam2["test"].value_counts()

import matplotlib.pyplot as plt
count_test = exam2["test"].value_counts()
count_test.plot.bar(rot=0)
plt.show()
plt.clf()

#200이상 : A
#100이상 : B
#100미만 : C
exam2["test2"] = np.where(exam2["total"] >= 200, "A",
                np.where(exam2["total"] >= 100, "B", "C"))
exam2.head()

exam2["test2"].isin(["A", "C"])
["A", "C"]





















