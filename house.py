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




