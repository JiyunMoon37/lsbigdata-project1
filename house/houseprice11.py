# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# 워킹 디렉토리 설정
import os
cwd=os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)
 
## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/house/train.csv")
house_test=pd.read_csv("./data/house/test.csv")
sub_df=pd.read_csv("./data/house/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()
## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()

# test 데이터 채우기
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]
for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()
## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()
house_train.shape
house_test.shape
train_n=len(house_train)
# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])
df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df
# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]
## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]
## test
test_x=test_df.drop("SalePrice", axis=1)


##
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
eln_model= ElasticNet()
rf_model= RandomForestRegressor(n_estimators=100)
# 그리드 서치 for ElasticNet
param_grid={
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}

grid_search=GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(train_x, train_y)

best_eln_model=grid_search.best_estimator_

# 그리드 서치 for RandomForests
param_grid={
    'max_depth': [3, 5, 7],
    'min_samples_split': [20, 10, 5],
    'min_samples_leaf': [5, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None]
}
grid_search=GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x, train_y)
grid_search.best_params_
best_rf_model=grid_search.best_estimator_


# 스택킹
y1_hat=best_eln_model.predict(train_x) # test 셋에 대한 집값
y2_hat=best_rf_model.predict(train_x) # test 셋에 대한 집값

train_x_stack=pd.DataFrame({
    'y1':y1_hat,
    'y2':y2_hat
})

from sklearn.linear_model import Ridge

rg_model = Ridge()
param_grid={
    'alpha': np.arange(0, 10, 0.01)
}
grid_search=GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x_stack, train_y)
grid_search.best_params_
blander_model=grid_search.best_estimator_

blander_model.coef_
blander_model.intercept_

pred_y_eln=best_eln_model.predict(test_x) # test 셋에 대한 집값
pred_y_rf=best_rf_model.predict(test_x) # test 셋에 대한 집값

test_x_stack=pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_rf
})

pred_y=blander_model.predict(test_x_stack)

# SalePrice 바꿔치기
# sub_df["SalePrice"] = pred_y
# sub_df
sub_df["SalePrice"] = pred_y
sub_df

## csv 파일로 내보내기
#sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)
#sub_df.to_csv("./data/houseprice/sample_submission11.csv", index=False)

##현주코드 
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
eln_model= ElasticNet()
rf_model= RandomForestRegressor(n_estimators=100)
# 그리드 서치 for ElasticNet
param_grid={
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}
grid_search=GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x, train_y)
best_eln_model=grid_search.best_estimator_

# 그리드 서치 for RandomForests
param_grid={
    'max_depth': [3, 5, 7],
    'min_samples_split': [20, 10, 5],
    'min_samples_leaf': [5, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None]
}
grid_search=GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x, train_y)
grid_search.best_params_
best_rf_model=grid_search.best_estimator_


# 스택킹
y1_hat=best_eln_model.predict(train_x) # test 셋에 대한 집값
y2_hat=best_rf_model.predict(train_x) # test 셋에 대한 집값

train_x_stack=pd.DataFrame({
    'y1':y1_hat,
    'y2':y2_hat
})

from sklearn.linear_model import Ridge

rg_model = Ridge()
param_grid={
    'alpha': np.arange(0, 10, 0.01)
}
grid_search=GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x_stack, train_y)
grid_search.best_params_
blander_model=grid_search.best_estimator_

blander_model.coef_
blander_model.intercept_

pred_y_eln=best_eln_model.predict(test_x) # test 셋에 대한 집값
pred_y_rf=best_rf_model.predict(test_x) # test 셋에 대한 집값

test_x_stack=pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_rf
})

pred_y=blander_model.predict(test_x_stack)

# SalePrice 바꿔치기
# sub_df["SalePrice"] = pred_y
# sub_df
sub_df["SalePrice"] = pred_y
sub_df

# # csv 파일로 내보내기
sub_df.to_csv("./data/house/sample_submission0911.csv", index=False)

#현주코드_엘라스틱넷 조정
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
eln_model= ElasticNet()
rf_model= RandomForestRegressor(n_estimators=100)
# 그리드 서치 for ElasticNet
param_grid={
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}
grid_search=GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x, train_y)
best_eln_model=grid_search.best_estimator_

# 그리드 서치 for RandomForests
param_grid={
    'max_depth': [3, 5, 7],
    'min_samples_split': [20, 10, 5],
    'min_samples_leaf': [5, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None]
}
grid_search=GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x, train_y)
grid_search.best_params_
best_rf_model=grid_search.best_estimator_


# 스택킹
y1_hat=best_eln_model.predict(train_x) # test 셋에 대한 집값
y2_hat=best_rf_model.predict(train_x) # test 셋에 대한 집값

train_x_stack=pd.DataFrame({
    'y1':y1_hat,
    'y2':y2_hat
})

from sklearn.linear_model import Ridge

rg_model = Ridge()
param_grid={
    'alpha': np.arange(0, 10, 0.01)
}
grid_search=GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search.fit(train_x_stack, train_y)
grid_search.best_params_
blander_model=grid_search.best_estimator_

blander_model.coef_
blander_model.intercept_

pred_y_eln=best_eln_model.predict(test_x) # test 셋에 대한 집값
pred_y_rf=best_rf_model.predict(test_x) # test 셋에 대한 집값

test_x_stack=pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_rf
})

pred_y=blander_model.predict(test_x_stack)

# SalePrice 바꿔치기
# sub_df["SalePrice"] = pred_y
# sub_df
sub_df["SalePrice"] = pred_y
sub_df

# # csv 파일로 내보내기
sub_df.to_csv("C:/Users/USER/Documents/LS빅데이터스쿨/ml/data/sample_submission2.csv", index=False)

#쌤 최종 커밋 (0911)
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 워킹 디렉토리 설정
import os
cwd=os.getcwd()
parent_dir = os.path.dirname(cwd)
os.chdir(parent_dir)

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

## NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기
house_train.isna().sum()
house_test.isna().sum()

## 숫자형 채우기
quantitative = house_train.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_train[col].fillna(house_train[col].mean(), inplace=True)
house_train[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)
house_train[qual_selected].isna().sum()


# test 데이터 채우기
## 숫자형 채우기
quantitative = house_test.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    house_test[col].fillna(house_train[col].mean(), inplace=True)
house_test[quant_selected].isna().sum()

## 범주형 채우기
qualitative = house_test.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_test[col].fillna("unknown", inplace=True)
house_test[qual_selected].isna().sum()


house_train.shape
house_test.shape
train_n=len(house_train)

# 통합 df 만들기 + 더미코딩
# house_test.select_dtypes(include=[int, float])

df = pd.concat([house_train, house_test], ignore_index=True)
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=np.log1p(train_df["SalePrice"])

## test
test_x=test_df.drop("SalePrice", axis=1)

# from sklearn.preprocessing import RobustScaler
# rs = RobustScaler()
# train_x=pd.DataFrame(rs.fit_transform(train_x), columns=train_x.columns) 
# test_x=pd.DataFrame(rs.fit_transform(test_x), columns=test_x.columns) 
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

eln_model= ElasticNet()
rf_model= RandomForestRegressor(n_estimators=500,
                                max_features="sqrt")

# 그리드 서치 for ElasticNet
param_grid={
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}
grid_search=GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5, verbose=1
)
grid_search.fit(train_x, train_y)
best_eln_model=grid_search.best_estimator_

# 그리드 서치 for RandomForests
param_grid={
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [5, 10, 20, 30],
    'min_samples_leaf': [5, 10, 20, 30]
    }
grid_search=GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5, verbose=1
)
grid_search.fit(train_x, train_y)
grid_search.best_params_
best_rf_model=grid_search.best_estimator_


# 스택킹
y1_hat=best_eln_model.predict(train_x) # test 셋에 대한 집값
y2_hat=best_rf_model.predict(train_x) # test 셋에 대한 집값

train_x_stack=pd.DataFrame({
    'y1':y1_hat,
    'y2':y2_hat
})

from sklearn.linear_model import Ridge

rg_model = Ridge()
param_grid={
    'alpha': np.arange(0, 10, 0.01)
}
grid_search=GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1
)
grid_search.fit(train_x_stack, train_y)
grid_search.best_params_
blander_model=grid_search.best_estimator_

blander_model.coef_
blander_model.intercept_

pred_y_eln=best_eln_model.predict(test_x) # test 셋에 대한 집값
pred_y_rf=best_rf_model.predict(test_x) # test 셋에 대한 집값

test_x_stack=pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_rf
})

pred_y=blander_model.predict(test_x_stack)

# SalePrice 바꿔치기
sub_df["SalePrice"] = np.expm1(pred_y)
sub_df

# # csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission_log1p.csv", index=False)

#용규오빠 코드 
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge, LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


# 데이터 로드
house_train = pd.read_csv("data/train.csv")
house_test = pd.read_csv("data/test.csv")
sub_df = pd.read_csv("data/sample_submission.csv")

# 결측값 처리
df = pd.concat([house_train, house_test], ignore_index=True)

# 범주형 결측값 처리
qualitative = df.select_dtypes(include=[object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]
for col in qual_selected:
    df[col].fillna("unknown", inplace=True)

# 숫자형 결측값 처리
quantitative = df.select_dtypes(include=[int, float])
fill_columns = quantitative.columns[quantitative.isna().sum() > 0]
train_columns = ['OverallQual', 'GrLivArea', 'TotalBsmtSF', '1stFlrSF', 'YearBuilt']

# 결측값 선형 회귀 함수
def nan_regression(df, fill_column, train_columns):
    train_data = df.dropna(subset=[fill_column])
    X_train = train_data[train_columns].dropna()
    y_train = train_data.loc[X_train.index, fill_column]
    
    model_in = LinearRegression()
    model_in.fit(X_train, y_train)
    
    test_data = df[df[fill_column].isna()]
    X_test = test_data[train_columns].fillna(X_train.mean())
    predicted_values = model_in.predict(X_test)
    
    df.loc[df[fill_column].isna(), fill_column] = predicted_values

for fill_column in fill_columns:
    nan_regression(df, fill_column, train_columns)

#임의.
df['YearBuilt_GrLivArea'] = df['YearBuilt'] * df['GrLivArea']
df['TotalBsmt_1stFlrSF'] = df['TotalBsmtSF'] * df['1stFlrSF']
df['OverallQual_GrLivArea'] = df['OverallQual'] * df['GrLivArea']


df = pd.get_dummies(df, columns=df.select_dtypes(include=[object]).columns, drop_first=True)
train_n = len(house_train)
train_df = df.iloc[:train_n,]
test_df = df.iloc[train_n:,]

train_x = train_df.drop(["SalePrice", "Id"], axis=1)
train_y = train_df["SalePrice"]
test_x = test_df.drop(["SalePrice", "Id"], axis=1)

# 데이터 스케일링 (StandardScaler 사용)
scaler = StandardScaler()
train_x_scaled = scaler.fit_transform(train_x)
test_x_scaled = scaler.transform(test_x)

# 첫 번째 층: ElasticNet, RandomForest, XGBoost

# ElasticNet
eln_model = ElasticNet()
param_grid_eln = {
    'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
}

grid_search_eln = GridSearchCV(
    estimator=eln_model,
    param_grid=param_grid_eln,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_eln.fit(train_x_scaled, train_y)
best_eln_model = grid_search_eln.best_estimator_

# RandomForest
rf_model = RandomForestRegressor(n_estimators=100)
param_grid_rf = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [20, 10, 5],
    'min_samples_leaf': [5, 10, 20, 30],
    'max_features': ['sqrt', 'log2', None]
}

grid_search_rf = GridSearchCV(
    estimator=rf_model,
    param_grid=param_grid_rf,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_rf.fit(train_x_scaled, train_y)
best_rf_model = grid_search_rf.best_estimator_

# XGBoost
xgb_model = XGBRegressor()
param_grid_xgb = {
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7]
}

grid_search_xgb = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid_xgb,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_xgb.fit(train_x_scaled, train_y)
best_xgb_model = grid_search_xgb.best_estimator_

# 첫 번째 층 모델 예측값
y1_hat = best_eln_model.predict(train_x_scaled)
y2_hat = best_rf_model.predict(train_x_scaled)
y3_hat = best_xgb_model.predict(train_x_scaled)


train_x_stack_1 = pd.DataFrame({
    'y1': y1_hat,
    'y2': y2_hat,
    'y3': y3_hat
})

# 두 번째 층: GradientBoosting, Ridge
# GradientBoostingRegressor 최적화(쓰라길래 써봄)
gb_model = GradientBoostingRegressor()
param_grid_gb = {
    'learning_rate': [0.01, 0.1],
    'n_estimators': [100, 200],
    'max_depth': [3, 5]
}

grid_search_gb = GridSearchCV(
    estimator=gb_model,
    param_grid=param_grid_gb,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_gb.fit(train_x_stack_1, train_y)
best_gb_model = grid_search_gb.best_estimator_

# Ridge 최적화
rg_model = Ridge()
param_grid_rg = {
    'alpha': np.arange(0, 10, 0.01)
}

grid_search_rg = GridSearchCV(
    estimator=rg_model,
    param_grid=param_grid_rg,
    scoring='neg_mean_squared_error',
    cv=5
)
grid_search_rg.fit(train_x_stack_1, train_y)
best_ridge_model = grid_search_rg.best_estimator_


pred_y_eln = best_eln_model.predict(test_x_scaled)
pred_y_rf = best_rf_model.predict(test_x_scaled)
pred_y_xgb = best_xgb_model.predict(test_x_scaled)

test_x_stack_1 = pd.DataFrame({
    'y1': pred_y_eln,
    'y2': pred_y_rf,
    'y3': pred_y_xgb
})

pred_y_gb = best_gb_model.predict(test_x_stack_1)
pred_y_ridge = best_ridge_model.predict(test_x_stack_1)

#2층 릿지로
pred_y_final = pred_y_ridge

# SalePrice 바꿔치기 및 저장
sub_df["SalePrice"] = pred_y_final
sub_df.to_csv("./data/sample_submission_stacked.csv", index=False)
