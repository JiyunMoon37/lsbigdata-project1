# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso  # Lasso 회귀 임포트
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

#워킹 디렉토리 설정
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

## 범주형 채우기
qualitative = house_train.select_dtypes(include = [object])
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    house_train[col].fillna("unknown", inplace=True)

# 통합 df 만들기 + 더미코딩
df = pd.concat([house_train, house_test], ignore_index=True)
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_n = len(house_train)
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

# Validation 셋(모의고사 셋) 만들기
np.random.seed(42)
val_index=np.random.choice(np.arange(train_n), size=438, replace=False)
val_index

# train => valid / train 데이터셋
valid_df=train_df.loc[val_index]  # 30%
train_df=train_df.drop(val_index) # 70%

## 이상치 탐색
train_df=train_df.query("GrLivArea <= 4500")

# x, y 나누기
# regex (Regular Expression, 정규방정식)
# ^ 시작을 의미, $ 끝남을 의미, | or를 의미
# selected_columns=train_df.filter(regex='^GrLivArea$|^GarageArea$|^Neighborhood_').columns

## train
train_x=train_df.drop("SalePrice", axis=1)
train_y=train_df["SalePrice"]

## valid
valid_x=valid_df.drop("SalePrice", axis=1)
valid_y=valid_df["SalePrice"]

## test
test_x=test_df.drop("SalePrice", axis=1)

# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

# 최적의 alpha 찾기
alphas = np.logspace(-4, 1, 50)  # 0.0001부터 10까지의 로그 스케일 alpha 값
param_grid = {'alpha': alphas}
lasso = Lasso()

grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, scoring='neg_mean_squared_error', cv=kf)
grid_search.fit(train_x, train_y)

# 최적의 alpha 값
best_alpha = grid_search.best_params_['alpha']
print(f"최적의 alpha 값: {best_alpha}")

# 모델 학습
model = Lasso(alpha=best_alpha)
model.fit(train_x, train_y)  # 자동으로 기울기, 절편 값을 구해줌

# 성능 측정 ()
y_hat=model.predict(valid_x)
rmse = np.sqrt(np.mean((valid_y-y_hat)**2))

# Test 데이터에 대한 예측
pred_y = model.predict(test_x)

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y

# csv 파일로 내보내기
sub_df.to_csv("./data/houseprice/sample_submission10.csv", index=False)