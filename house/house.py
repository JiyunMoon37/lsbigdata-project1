# import numpy as np
import pandas as pd 

house_df = pd.read_csv("./data/house/train.csv")
house_df.shape
house_df.head()
house_df.info()

price_mean = house_df["SalePrice"].mean()
price_mean

# year_mean = house_df["SalePrice"].mean()
# year_mean



#0729 
#house_df의 
house_df = pd.read_csv("./data/house/train.csv")
house_df = house_df.groupby('YearBuilt',as_index = False)\
                    .agg(price_mean = ('SalePrice','mean')) #그룹을 나눠 요약을 한다. 
house_df

test_df = pd.read_csv("./data/house/test.csv")
merge_df = pd.merge(test_df, house_df, how = 'left', on = 'YearBuilt')


sub_df = pd.read_csv("./data/house/sample_submission.csv")
sub_df

sub_df["SalePrice"] = price_mean
sub_df.to_csv("sample_submission2.csv", index = False) 
---
#쌤 풀이

house_train = pd.read_csv("./data/house/train.csv")
house_train = house_train[["Id", "YearBuilt", "SalePrice"]]
house_train.info()

#연도별 평균
house_mean = house_train.groupby('YearBuilt', as_index = False)\
                    .agg(mean_year = ('SalePrice','mean')) #그룹을 나눠 요약을 한다. 
house_mean

house_test = pd.read_csv("./data/house/test.csv")
house_test = house_test[["Id", "YearBuilt"]]
house_test

house_test = pd.merge(house_test, house_mean, how = "left", on = "YearBuilt")
house_test = house_test.rename(columns = {"mean_year" : "Saleprice"})
house_test

house_test["Saleprice"].isna().sum() #9개가 값이 없다.
#비어있는 테스트 세트 집들 확인 
house_test.loc[house_test["Saleprice"].isna()] #9개 값이 무엇인지 알기 
house_mean = house_train["SalePrice"].mean()
house_test["Saleprice"] = house_test["Saleprice"].fillna(house_mean)
house_test


#sub 데이터 불러오기 
sub_df = pd.read_csv("./data/house/sample_submission.csv")
sub_df

#SalePrice 바꿔치기 
sub_df["SalePrice"] = house_test["Saleprice"]
sub_df

sub_df.to_csv("./data/house/sample_submission2.csv", index = False) 


------
house_train = house_train[["Id", "YearBuilt", "SalePrice"]]
house_train.info()

#연도별 평균
house_mean = house_train.groupby('YearBuilt', as_index = False)\
                    .agg(mean_year = ('SalePrice','mean')) #그룹을 나눠 요약을 한다. 
house_mean

house_test = pd.read_csv("./data/house/test.csv")
house_test = house_test[["Id", "YearBuilt"]]
house_test

house_test = pd.merge(house_test, house_mean, how = "left", on = "YearBuilt")
house_test = house_test.rename(columns = {"mean_year" : "Saleprice"})
house_test

house_test["Saleprice"].isna().sum() #9개가 값이 없다.
#비어있는 테스트 세트 집들 확인 
house_test.loc[house_test["Saleprice"].isna()] #9개 값이 무엇인지 알기 
house_mean = house_train["SalePrice"].mean()
house_test["Saleprice"] = house_test["Saleprice"].fillna(house_mean)
house_test


#sub 데이터 불러오기 
sub_df = pd.read_csv("./data/house/sample_submission.csv")
sub_df

#SalePrice 바꿔치기 
sub_df["SalePrice"] = house_test["Saleprice"]
sub_df

sub_df.to_csv("./data/house/sample_submission2.csv", index = False) 

#0731 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import t 

house_df = pd.read_csv("./data/house/train.csv")
house_df

house_df.info()

house_df = house_df[["BldgType", "HouseStyle"]]
house_df["BldgType"].isna().sum() #결측치 알아보기 ->결측치 없음 
house_df["HouseStyle"].isna().sum() #결측치 알아보기 -> 결측치 없음 

house = house_df.copy()
house.describe()

house_bt = house["BldgType"].value_counts()
house_bt_df = house_bt.reset_index() #데이터프레임으로 불러오기 
house_bt_df.columns = ['BldgType', 'Count']  # 열 이름 지정
sns.barplot(data=house_bt_df, x = 'BldgType', y = 'Count', hue = 'BldgType')

plt.show()
plt.clf()

sex_income = house_bt_df.groupby("sex", as_index = False) \
                       .agg(mean_income = ("income", "mean"))
sex_income



# 
# #값 바꾸려고 함 
# house['BldgType'] = house['BldgType'].replace({
#     '2FmCon': '2Fam',
#     'Duplx': '2Fam',
#     'TwnhsE': 'Town',
#     'TwnhsI': 'Town'
# })

# house.loc[house['BldgType'] == 'TwnhsE', 'BldgType'] = 'Town'
# house.loc[house['BldgType'] == 'TwnhsI', 'BldgType'] = 'Town'
# 
# house.loc[house['BldgType'] == '2FmCon', 'BldgType'] = '2Fam'
# house.loc[house['BldgType'] == 'Duplx', 'BldgType'] = '2Fam'
# 
# house["BldgType"].value_counts()


#housestyle별 개수 
house_hs =house["HouseStyle"].value_counts()
house_hs_df = house_hs.reset_index() #데이터프레임으로 불러오기 
house_hs_df.columns = ['HouseStyle', 'Count']  # 열 이름 지정
sns.barplot(data=house_hs_df, x = 'HouseStyle', y = 'Count', hue = 'HouseStyle')

plt.show()
plt.clf()
















