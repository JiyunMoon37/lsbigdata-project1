import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error
import statsmodels.api as sm

import os

# !pip install statsmodels

os.getcwd()
os.chdir('c://Users//USER//Documents//LS 빅데이터 스쿨//lsbigdata-project1')

train_df = pd.read_csv('./blueberry/train.csv')
train_df.drop(columns=('id'))
test_df = pd.read_csv('./blueberry/test.csv')
submission = pd.read_csv('./blueberry/sample_submission.csv')

blue_train = pd.read_csv("./data/blueberry/train.csv")
blue_test = pd.read_csv("./data/blueberry/test.csv")
sub_df = pd.read_csv("./data/blueberry/sample_submission.csv")

# 데이터 파악 : 다 수치 변수임
train_df.info()
test_df.info()


# 결측값 없음
nan_train = train_df.isna().sum()
nan_train[nan_train>0]

nan_test = test_df.isna().sum()
nan_test[nan_test>0]


# 데이터 개수
len(train_df)
len(test_df)

train_x = train_df.drop(columns=('yield'))
train_y = train_df['yield']

test_x = test_df.copy()


# 일반회귀 모든 변수
model = LinearRegression()
model.fit(train_x, train_y)
train_y_pred = model.predict(train_x)

print("train MAE : ", np.abs(train_y_pred - train_y).mean() )

test_y_pred = model.predict(test_x)

submission['yield'] = test_y_pred
submission.to_csv('./blueberry/all_linearregression.csv', index=False)