import pandas as pd 
import numpy as np

#데이터 전처리 함수수
#query() 행 추출
#df[] 열 추출
#sort_values() 정렬
#groupby() 집단별로 나누기
#assign() 변수추가
#agg()
#merge()
#concat()

exam = pd.read_csv("data/exam.csv")
#조건에 맞는 행을 걸래는 .query()
exam.query("nclass == 1")
#exam[exam["nclass"] == 1]

#p136
#수학 점수가 50점을 초과한 경우
exam.query('math > 50')

#수학 점수가 50점 미만인 경우
exam.query('math < 50')

#영어 점수가 50점 이상인 경우
exam.query('english >= 50')

#영어 점수가 80점 이하인 경우
exam.query('english <= 80')

#p138
#1반이면서 수학 점수가 50미만인 경우
exam.query('nclass == 1 & math >= 50')

#2반이면서 영어 점수가 80점 이상인 경우
exam.query('nclass == 2 & english >= 80')

#수학 점수가 90점 이상이거나 영어 점수가 90점 이상인 경우 / | 대신 or 써도 됨. 
exam.query('math >= 90 | english >= 90')

#영어 점수가 90점 미만이거나 과학 점수가 50점 미만인 경우
exam.query('english < 90 | science < 50')

#1, 3, 5반에 해당하면 추출
exam.query('nclass == 1 | nclass ==3 | nclass ==5')

#1, 3, 5반에 해당하면 추출
exam.query('nclass in [1, 3, 5]')
exam.query('nclass not in [2, 4]')
# exam[~exam["nclass"].isin[1,2]]

#p145
exam["nclass"]
exam[["nclass"]]
exam[["id", "nclass"]]
exam.drop(columns = "math")
exam.drop(columns = ["math", "english"])
exam 

exam.query("nclass == 1")[["math", "english"]]
exam.query("nclass == 1") \
    [["math", "english"]] \
    .head()

#정렬하기
exam.sort_values("math")
exam.sort_values("math", ascending = False)
exam.sort_values(["nclass", "english"], ascending = [True, False])

#변수 추가
exam = exam.assign(
    total = exam["math"] +exam["english"] +exam["science"],
    mean = (exam["math"] +exam["english"] +exam["science"]) / 3
    ) \
    .sort_values("total", ascending = False)
exam.head()

#lamda 함수 사용하기 
exam2 = pd.read_csv("data/exam.csv")

exam2 = exam.assign(
    total = lamda x : x["math"] +x["english"] +x["science"],
    mean = lamda x : (x["total"]) / 3
    ) \
    .sort_values("total", ascending = False)
exam2.head()


#그룹을 나눠 요약을 하는 .groupby() + .agg() 콤보 
exam2.agg(mean_math = ("math", "mean"))
exam2.groupby("nclass") \
    .agg(mean_math = ("math", "mean"))

exam2.agg(mean_math = ("math", "mean"))

exam2.groupby("nclass") \
    .agg(mean_math = ("math", "mean"),
        sum_math = ("math", "sum"),
        median_math = ("math", "median"),
        n = ("nclass", "count"))

exam2.groupby("nclass") \
    .agg(mean_math = ("math", "mean"),
       mean_eng = ("english", "sum"),
        mean_sci = ("science", "median"),
        )

#p165_pandas 함수 조합하기 
#Q. 제조 회사별로 'suv' 자동차의 도시 및 고속도로 합산 연비 평균을 구해 내림차순으로 정렬하고, 1~5위까지 출력하기 

import pydataset

import pandas as pd 

df = pydataset.data("mpg")
df

test1 = pd.DataFrame({"id"      : [1, 2, 3, 4, 5], 
                      "midterm" : [60, 80, 70, 90, 85]})
    
test2 = pd.DataFrame({"id"      : [1, 2, 3, 4, 5], 
                      "midterm" : [70, 83, 65, 95, 80]})
                      
test1 
test2

total = pd.merge(test1, test2, how="left", on = "id")
total

#left join
test1 = pd.DataFrame({"id"      : [1, 2, 3, 4, 5], 
                      "midterm" : [60, 80, 70, 90, 85]})
    
test2 = pd.DataFrame({"id"      : [1, 2, 3, 40, 5], 
                      "midterm" : [70, 83, 65, 95, 80]})

total = pd.merge(test1, test2, how="left", on = "id")
total

#right join
test1 = pd.DataFrame({"id"      : [1, 2, 3, 4, 5], 
                      "midterm" : [60, 80, 70, 90, 85]})
    
test2 = pd.DataFrame({"id"      : [1, 2, 3, 40, 5], 
                      "midterm" : [70, 83, 65, 95, 80]})

total = pd.merge(test1, test2, how="right", on = "id")
total

#inner join
total = pd.merge(test1, test2, how="inner", on = "id")
total

#outer join
total = pd.merge(test1, test2, how="outer", on = "id")
total

#책 p169
name = pd.DataFrame({"nclass" : [1, 2, 3, 4, 5],
                     "teacher" : ["kim", "lee", "park", "choi", "jung"]})
name 

exam = pd.read_csv("data/exam.csv")
exam

exam_new = pd.merge(name, exam, how = "left", on = "nclass")
exam_new

#데이터 세로로 쌓기 
score1 = pd.DataFrame({"id"      : [1, 2, 3, 4, 5], 
                      "score" : [60, 80, 70, 90, 85]})
    
score2 = pd.DataFrame({"id"      : [6, 7, 8, 9, 10], 
                      "score" : [70, 83, 65, 95, 80]})
score_all = pd.concat([score1, score2])   
score_all


































































