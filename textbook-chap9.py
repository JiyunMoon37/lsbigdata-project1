#!pip install pyreadstat
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

raw_welfare = pd.read_spss('./data/koweps/Koweps_hpwc14_2019_beta2.sav')
raw_welfare

welfare = raw_welfare.copy()
welfare.shape
welfare.describe()

welfare = welfare.rename(
    columns = {
        "h14_g3" : "sex",
        "h14_g4" : "birth",
        "h14_g10" : "marriage_type",
        "h14_g11" : "religion",
        "p1402_8aq1" : "income",
        "h14_eco9" : "code_job",
        "h14_reg7" : "code_region"
    }
)

welfare = welfare[["sex", "birth", "marriage_type", "religion", "income", "code_job", "code_region"]]
welfare.shape 

#4교시 
welfare["sex"].dtypes
welfare["sex"].value_counts()
welfare["sex"].isna().sum()

welfare["sex"] = np.where(welfare["sex"]==1,
                          'male', 'female')
welfare

welfare["income"].describe() #요약 통계량 구하기
welfare["income"].isna().sum()

sex_income = welfare.dropna(subset = "income") \
                    .groupby("sex", as_index = False) \
                    .agg(mean_income = ("income", "mean"))
sex_income 

import seaborn as sns
sns.barplot(data=sex_income, x = "sex", y = "mean_income", 
            hue = "sex")
            
plt.show()
plt.clf() 

welfare["birth"].describe()
sns.histplot(data=welfare, x = "birth")
plt.show()
plt.clf() 

welfare["birth"].isna().sum()


welfare["birth"].isna().sum()



welfare = welfare.assign(age = 2019 - welfare["birth"] + 1)
welfare["age"]
sns.histplot(data = welfare, x = "age")
plt.show()
plt.clf()

age_income = welfare.dropna(subset = 'income') \
                    .groupby('age', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))
sns.lineplot(data=age_income, x = "age", y = "mean_income")
plt.show()
plt.clf()

#나이별 income column에서 na 개수 세기!
welfare["income"].isna().sum()

my_df = welfare.assign(income_na=welfare["income"].isna()) \
                       .groupby("age", as_index=False) \
                       .agg(n = ("income_na", "sum")) 

sns.barplot(data = my_df, x = "age", y = "n")
plt.show()
plt.clf()

#p240
#나이 변수 살펴보기 
welfare['age'].head()

#연령대 변수 만들기
welfare = welfare.assign(ageg = np.where(welfare['age'] < 30, 'young',
                                np.where(welfare['age'] <= 59, 'middle',
                                                                'old')))

#빈도 구하기
welfare['ageg'].value_counts()

#빈도 막대 그래프 만들기
sns.countplot(data = welfare, x = 'ageg')

#연령대별 월급 평균표 만들기
ageg_income = welfare.dropna(subset = 'income')\
                    .groupby('ageg', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))
                    
#막대그래프 만들기
sns.barplot(data = ageg_income, x = 'ageg', y = 'mean_income')

#막대 정렬하기
sns.barplot(data = ageg_income, x = 'ageg', y = 'mean_income',
            order = ['young', 'middle', 'old'])

plt.show()
plt.clf()


#연령대 별로 나누기 / 0~9 / 10~19 / 20~29 / 
#cut -> 특정 벡터가 있을 때 기준값을 기준으로 cut 해주는 함수 

#내가 짠 코드 
bins = [0, 10, 20, 30, 40, np.inf]
labels = ['baby', 'teen', 'semi_adult' ,'adult' ,'old_adult']

welfare = welfare.assign(age_group = pd.cut(welfare['age'], bins = bins, labels=labels, right=False))
welfare

#나이대별 명수 
age_group_counts = welfare['age_group'].value_counts()

sns.barplot(x=age_group_counts.index, y =age_group_counts.values)
plt.show()
plt.clf()




#나이대별 소득 
#age_group_counts = welfare['age_group'].value_counts()
bins = [0, 10, 20, 30, 40, np.inf]
labels = ['baby', 'teen', 'semi_adult' ,'adult' ,'old_adult']

welfare = welfare.assign(age_group = pd.cut(welfare['age'], bins = bins, labels=labels, right=False))
welfare

sns.barplot(x='age_group', y ='mean_income', data = ageg_income)
plt.show()
plt.clf()


vec_x = np.random.randint(0, 100, 50)

np.arange(13)*10 - 1
pd.cut(vec_x, bins=bin_cut)


#나이대별 수입 분석
#cut
bin_cut = np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare = welfare.assign(age_group = pd.cut(welfare['age'], 
                  bins = bin_cut,
                  labels = (np.arange(12) * 10).astype(str) + "대"))

welfare["age_group"]
welfare

age_income = welfare.dropna(subset = 'income') \
                    .groupby('age_group', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))

age_income
sns.barplot(data=age_income, x = "age_group", y = "mean_income")
plt.show()
plt.clf()
welfare



#7교시 
#p244
#판다스 데이터 프레임을 다룰 때, 변수의 타입이
#카테고리로 설정되어 있는 경우, groupby+agg 콤보
#안먹힘, 그래서 object 타입으로 바꿔준 후 수행 
welfare["age_group"] = welfare["age_group"].astype("object")

sex_age_income = welfare.dropna(subset = "income") \
    .groupby(["age_group", "sex"], as_index = False) \
    .agg(top4per_income = ("income", 
                            lambda x: np.quantile(x, q=0.96)))
sex_age_income

sns.barplot(data = sex_age_income, 
            x = "age_group", y = "mean_income",
            hue = "sex")
plt.show()
plt.clf()         

#연령대별, 성별 상위 4% 수입 찾아보세요!
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

welfare

np.mean(np.arange(10)) #벡터에 mean 씌워서 값이 나옴. 
quantile(vec, 0.95)



#그래프로 나타내기 
sns.barplot(data=age_income, x = "age_group", y = "mean_income")
plt.show()
plt.clf()
welfare

-----
welfare["age_group"] = welfare["age_group"].astype("object")

sex_age_income = welfare.dropna(subset = "income") \
    .groupby(["age_group", "sex"], as_index = False) \
    .agg(top4per_income = ("income", 
                            lambda x: np.quantile(x, q=0.96)))
sex_age_income

#gpt
#사용자 정의 함수
def custom_mean(series, dropna = True):
    if dropna :
        return print(series, "hi")
    else :
        return print(series, "hello")
    
#그룹화 및 사용자 정의 함수 적용
sex_age_income = welfare.dropna(subset = ["age_group", "sex"]) \
                        .groupby(["age_group", "sex"], as_index = False) \
                        .agg(mean_income = ("income", custom_mean))
                        
sex_age_income = welfare.dropna(subset = ["age_group", "sex"]) \
                        .groupby(["age_group", "sex"], as_index = False) \
                        .agg(mean_income = ("income", lambda x: custom_mean(x, dropna=False)))
x = np.arange(10)
np.quantile(x, q=0.5)


#결과 출력 
print(sex_age_income) 

# 연령대별, 성별 상위 4% 수입 찾아보기
sex_age_income = welfare.dropna(subset = "income") \
    .groupby(["age_group", "sex"], as_index = False) \
    .agg(top4per_income = ("income", lambda x : np.quantile(x, q=0.96)))
sex_age_income


sex_age_income = welfare.dropna(subset = "income") \
    .groupby(["age_group", "sex"], as_index = False) \
    .agg(top4per_income = ("income", lambda x : np.quantile(x, q=0.96)))
sex_age_income


#07315교시
#!pip install pyreadstat
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import t 

#어제 복습 
#나이대별 수입 분석
#cut
bin_cut = np.array([0, 9, 19, 29, 39, 49, 59, 69, 79, 89, 99, 109, 119])
welfare = welfare.assign(age_group = pd.cut(welfare['age'], 
                  bins = bin_cut,
                  labels = (np.arange(12) * 10).astype(str) + "대"))

welfare["age_group"]
welfare

age_income = welfare.dropna(subset = 'income') \
                    .groupby('age_group', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))

age_income
sns.barplot(data=age_income, x = "age_group", y = "mean_income")
plt.show()
plt.clf()

sex_age_income = welfare.dropna(subset = "income") \
    .groupby(["age_group", "sex"], as_index = False) \
    .agg(top4per_income = ("income", 
                            lambda x: np.quantile(x, q=0.96)))
sex_age_income #나이대 별 상위 4% 수입을 알 수 있다. 

sns.barplot(data = sex_age_income, 
            x = "age_group", y = "top4per_income",
            hue = "sex")
plt.show()
plt.clf()

#남규님, 지원님 코드
sex_income2 = welfare.dropna(subset = 'income') \
                    .groupby('sex', as_index = False)[['income']] \
                    .agg(['mean', 'std'])
#[]쓰면 시리즈, []2개 쓰면 데이터 프레임 
#agg를 쓰면 mean, std 둘 다 뽑을 수 있다. 

#6교시
#p248
welfare["code_job"]
welfare["code_job"].value_counts()

list_job = pd.read_excel("./data/koweps/Koweps_Codebook_2019.xlsx", sheet_name = "직종코드")
list_job.head()

welfare = welfare.merge(list_job, how = "left", on= "code_job")
#welfare.dropna(subset = 'code_job')[['code_job', 'job']].head()
welfare.dropna(subset=["job", "income"])[["income", "job"]]

job_income = welfare.dropna(subset = ['job', 'income']) \
                    .groupby('job', as_index = False) \
                    .agg(mean_income = ('income', 'mean'))
                    
job_income.head()

top10 = job_income.sort_values('mean_income', ascending = False).head(10)
top10

import matplotlib.pyplot as plt
plt.rcParams.update({'font.family' : 'Malgun Gothic'})

sns.barplot(data = top10, y = 'job', x = 'mean_income', palette='Set2', hue = 'job')
#plt.tighy_layout()
plt.show()
plt.clf()

#p255
#남자 직업 빈도 상위 10개 추출 
job_male = welfare.dropna(subset = 'job') \
                    .query('sex == "male"') \
                    .groupby('job', as_index = False) \
                    .agg(n = ('job', 'count')) \
                    .sort_values('n', ascending = False)\
                    .head(10)
job_male 

#여자 직업 빈도 상위 10개 추출 
job_female = welfare.dropna(subset = 'job') \
                    .query('sex == "female"') \
                    .groupby('job', as_index = False) \
                    .agg(n = ('job', 'count')) \
                    .sort_values('n', ascending = False)\
                    .head(10)
job_female 

#p263 
welfare.info()
welfare["marriage_type"]

df = welfare.query("marriage_type != 5") \
            .groupby("religion", as_index = False) \
            ["marriage_type"] \
            .value_counts(normalize = True)
df

df.query("marriage_type == 1") \
    .assign(proportion=df["proportion"]*100).round(1) 








































