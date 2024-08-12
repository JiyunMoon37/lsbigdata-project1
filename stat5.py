#패키지 불러오기
import numpy as np
import pandas as pd 
from scipy.stats import t
from scipy.stats import norm

tab3 = pd.read_csv("./data/tab3.csv")
tab3 

tab1 = pd.DataFrame({"id" : np.arange(1,13),
                     "score" : tab3['score']})
tab1

tab2 = pd.DataFrame({"id" : np.arange(1,13),
                     "score" : tab3['score']},
                     "gender" : )

tab2 = tab1.assign(gender=["female"]*7 + ["male"]*5)
tab2 

#1표본 t검정 (그룹 1개)
#귀무가설 vs. 대립가설
#H0 : mu = 10 vs. Ha : mu != 10 
#유의수준 5%로 설정 
from scipy.stats import ttest_1samp

result = ttest_1samp(tab1["score"], popmean = 10, alternative = 'two-sided') #모평균 : popmean
print("t-statistic:", t_statistic)

t_value = result[0] #t검정통계량 
p_value = result[1] #양쪽을 더한 값, 유의확률 (p-value) 
tab1["score"].mean() #11.53이 나온다. 

result.pvalue
result.statistic #t_value 
result.df

#귀무가설이 참(mu=10)일때, 11.53이 관찰될 확률이 6.48%이므로,(모평균이 10이니까 나올만한겨)
#이것은 우리가 생각하는 보기 힘들다고 판단하는 기준인
#0.05 (유의수준)(관찰이 안된다는 수준)보다 크므로, 귀무가설이 거짓이라 판단하기 힘들다. 

#유의확률 0.0648이 유의수준 0.05보다 크므로
#귀무가설을 기각하지 못한다. 


ci = result.confidence_interval(confidence_level = 0.95)
ci[0]
ci[1]

#2표본 t검정 (그룹2)
##귀무가설 vs. 대립가설
##H0 : mu_m = mu_f vs. Ha : mu_m > mu_f
##H0 : mu_before = mu_after vs. Ha: mu_after > mu_before
##H0 : mu_d = 0 vs. Ha: mu_d > 0
##mu_d = mu_after - mu_before 
##유의수준 1%로 설정, 두 그룹 분산 같다고 가정한다. 
tab2 
f_tab2 = tab2[tab2["gender"] == "female"]
m_tab2 = tab2[tab2["gender"] == "male"]

from scipy.stats import ttest_ind 

#alternative = "less"의 의미는 대립가설이
#첫 번째 입력그룹의 평균이 두 번째 입력그룹의 평균보다 작다고 설정된 경우를 나타냄. 
result = ttest_ind(f_tab2["score"], m_tab2["score"], 
          alternative = 'less', equal_var = True)
#female 값이 먼저 나왔는데 그 값이 male보다 작으므로 alternative는 less 

result.statistic
result.pvalue
ci = result.confidence_interval(0.95)
ci[0]
ci[1]

#분산 같은 경우 : 독립 2표본 t검정
#분산 다를 경우 : 웰치스 t 검정 


result = ttest_1samp(tab1["score"], popmean = 10, alternative = 'two-sided') #모평균 : popmean
print("t-statistic:", t_statistic)

t_value = result[0] #t검정통계량 
p_value = result[1] #양쪽을 더한 값, 유의확률 (p-value) 
tab1["score"].mean() #11.53이 나온다. 

result.pvalue
result.statistic #t_value 
result.df

#대응표본 t 검정 (짝지을 수 있는 표본)
tab3
tab3_data = tab3.pivot_table(index = 'id',
                            columns = 'group', 
                            values='score').reset_index()
                            
#id는 그대로 있다. #테이블 값을 채우는 것은 score
tab3_data['score_diff'] = tab3_data['after'] - tab3_data['before']
test3_data = tab3_data[['score_diff']]
test3_data 

long_form = tab3_data.reset_index().melt(id_vars = 'id', value_vars = ['before','after'])


from scipy.stats import ttest_1samp

result = ttest_1samp(test3_data["score_diff"], popmean = 0, alternative = 'greater') #모평균 : popmean
t_value = result[0] #t검정통계량 
p_value = result[1] #양쪽을 더한 값, 유의확률 (p-value) 
t_value : p_value 


##0806 4교시
#melt 코드 
melt(id_vars = 'id',
     value_var = ['A','B'],
     var_name = 'group',
     value_name = 'score')


long_form = tab3_data.melt(
    id_vars = 'id',
    value_var = ['A','B'],
    var_name = 'group',
    value_name = 'score'
    )
    
#연습1
df = pd.DataFrame({"id" : [1, 2, 3],
                   "A" : [10, 20, 30],
                   "B" : [40, 50, 60]})
                   
df_long = df.melt(id_vars = "id",
                  value_vars = ["A", "B"],
                  var_name = "group",
                  value_name = "score")

df_long.pivot_table(
    columns = "group", 
    values = "score"
    )

df_long.pivot_table(
    columns = "group", 
    values = "score",
    aggfunc = "mean"
    )  
#최댓값 하고 싶으면 aggfunc = "max"




#연습2
#!pip install seaborn
import seaborn as sns
tips = sns.load_dataset('tips')
tips

#요일별로 펼치고 싶은 경우
#day만 빼고 싶을 때
tips.columns.delete(4)




tips.reset_index(drop = False) \
    .pivot_table(
        index = ["index", "total_bill"], 
        columns = "day",
        values = "tip").reset_index() 

tips.pivot(colums = "day", 
           volumes = "tip").reset_index() 














