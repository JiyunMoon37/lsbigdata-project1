#펭귄 데이터 부리길이 예측 모형 만들기
#엘라스틱 넷&디시전트리 회귀모델 사용
#모든 변수 자유롭게 사용!
#종속변수 : bill_length_mm

# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from palmerpenguins import load_penguins
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

# 데이터 로드
data = load_penguins()

# NaN 채우기
data.dropna(inplace=True)  # 결측치 제거

# 독립 변수와 종속 변수 나누기
X = data.drop(columns=['bill_length_mm'])
y = data['bill_length_mm']

# 범주형 변수를 더미 변수로 변환
X = pd.get_dummies(X, drop_first=True)

# train/test 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 엘라스틱 넷 모델 학습 및 하이퍼파라미터 튜닝
elastic_net = ElasticNet()
param_grid_en = {
    'alpha': [0.1, 1.0, 10.0, 100.0],
    'l1_ratio': [0, 0.1, 0.5, 1.0]
}

grid_search_en = GridSearchCV(
    estimator=elastic_net,
    param_grid=param_grid_en,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search_en.fit(X_train, y_train)

# 최적의 하이퍼파라미터
best_params_en = grid_search_en.best_params_
best_model_en = grid_search_en.best_estimator_

# 예측 및 성능 측정
y_pred_en = best_model_en.predict(X_test)
mse_en = mean_squared_error(y_test, y_pred_en)
rmse_en = np.sqrt(mse_en)

# 결정 트리 모델 학습
decision_tree = DecisionTreeRegressor()
decision_tree.fit(X_train, y_train)

# 예측 및 성능 측정
y_pred_dt = decision_tree.predict(X_test)
mse_dt = mean_squared_error(y_test, y_pred_dt)
rmse_dt = np.sqrt(mse_dt)

# 결과 출력
print(f'엘라스틱 넷 최적 하이퍼파라미터: {best_params_en}')
print(f'엘라스틱 넷 RMSE: {rmse_en}')
print(f'결정 트리 RMSE: {rmse_dt}')

# 예측 결과 시각화
plt.rc('font', family='Malgun Gothic')
plt.scatter(y_test, y_pred_en, label='ElasticNet Predictions', alpha=0.5)
plt.scatter(y_test, y_pred_dt, label='Decision Tree Predictions', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.xlabel('실제 부리 길이')
plt.ylabel('예측 부리 길이')
plt.title('부리 길이 예측')
plt.legend()
plt.show()

#다른 팀 코드
# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디시젼트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수 : bil_length_mm

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins=load_penguins()
penguins.head()

## Nan 채우기
quantitative = penguins.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    penguins[col].fillna(penguins[col].mean(), inplace=True)
penguins[quant_selected].isna().sum()

## 범주형 채우기
qualitative = penguins.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    penguins[col].fillna(penguins[col].mode()[0], inplace=True)
penguins[qual_selected].isna().sum()

df = penguins
df = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first = True
)
df

x=df.drop("bill_length_mm", axis=1)
y=df[['bill_length_mm']]
x
y

## 모델 생성
from sklearn.linear_model import ElasticNet
model = ElasticNet()

## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

param_grid={
    'alpha': np.arange(0, 0.2, 0.01),
    'l1_ratio': np.arange(0.8, 1, 0.01)
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(x,y)

grid_search.best_params_ #alpha=0.19, l1_ratio=0.99
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

##
# 모델 생성
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)
param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1),
    'min_samples_leaf' : np.arange(10, 30, 1)
}

# 하이퍼파라미터 튜닝
grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(x,y)

grid_search.best_params_ #8, 22
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

#0904 6교시
# 펭귄 데이터 부리길이 예측 모형 만들기
# 엘라스틱 넷 & 디시젼트리 회귀모델 사용
# 모든 변수 자유롭게 사용!
# 종속변수: bill_length_mm

import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
from sklearn.preprocessing import OneHotEncoder

penguins = load_penguins()
penguins.head()

## Nan 채우기
quantitative = penguins.select_dtypes(include = [int, float])
quantitative.isna().sum()
quant_selected = quantitative.columns[quantitative.isna().sum() > 0]

for col in quant_selected:
    penguins[col].fillna(penguins[col].mean(), inplace=True)
penguins[quant_selected].isna().sum()

## 범주형 채우기
qualitative = penguins.select_dtypes(include = [object])
qualitative.isna().sum()
qual_selected = qualitative.columns[qualitative.isna().sum() > 0]

for col in qual_selected:
    penguins[col].fillna(penguins[col].mode()[0], inplace=True)
penguins[qual_selected].isna().sum()

df = penguins
df = pd.get_dummies(
    df,
    columns = df.select_dtypes(include=[object]).columns,
    drop_first = True
)
df

x=df.drop("bill_length_mm", axis=1)
y=df[['bill_length_mm']]
x
y

## 모델 생성
from sklearn.linear_model import ElasticNet
model = ElasticNet()

## 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV

param_grid={
    'alpha': np.arange(0, 0.2, 0.01),
    'l1_ratio': np.arange(0.8, 1, 0.01)
}

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(x,y)

grid_search.best_params_ #alpha=0.19, l1_ratio=0.99
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_
##
# 모델 생성
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=42)
param_grid={
    'max_depth': np.arange(7, 20, 1),
    'min_samples_split': np.arange(10, 30, 1)
}

# 하이퍼파라미터 튜닝

grid_search=GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5
)

grid_search.fit(x,y)

grid_search.best_params_ #8, 22
grid_search.cv_results_
grid_search.best_score_
best_model=grid_search.best_estimator_

model = DecisionTreeRegressor(random_state=42,
                              max_depth=2,
                              min_samples_split=22)
model.fit(x,y)

from sklearn import tree
tree.plot_tree(model)