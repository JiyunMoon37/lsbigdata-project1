#####################################################################################
## 이번에는 블루베리 데이터로 해보기.
#####################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, mean_squared_error

### 데이터 불러오기
blue_train = pd.read_csv("./data/blueberry/train.csv")
blue_test = pd.read_csv("./data/blueberry/test.csv")
sub_df = pd.read_csv("./data/blueberry/sample_submission.csv")

### 데이터 알아보기?
blue_train.shape #(15289, 18)
blue_test.shape #(15289, 18)
train_n=len(blue_train)

#### train 데이터 전처리 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#### 데이터 전처리

### NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기

# character
# numeric
## 숫자형 채우기
numeric_train = blue_train.select_dtypes(include = [int, float]) # 숫자형 컬럼만 선택해서 데이터프레임으로 가져오기.
numeric_train.isna().sum() # NaN 값이 있는지 확인. #없당 

# 이상치 처리: honeybee가 1보다 큰 데이터 삭제
blue_train = blue_train[blue_train['honeybee'] <= 1]


#### train 데이터 전처리 끝 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@Series([], dtype: float64)


#### test 데이터 전처리 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
numeric_test = blue_test.select_dtypes(include = [int, float])
numeric_test.isna().sum() # NaN 값이 있는지 확인. #없당 

# 이상치 처리: honeybee가 1보다 큰 데이터 삭제
blue_test = blue_test[blue_test['honeybee'] <= 1]

#### test 데이터 전처리 끝 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


### 통합 df 만들기 + 더미코딩
## 통합 df 만들기
df = pd.concat([blue_train, blue_test], ignore_index=True)

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## x / y 나누기
# train
train_x = train_df[["fruitset", "fruitmass", "seeds"]]
train_y=train_df["yield"]

test_x = test_df[["fruitset", "fruitmass", "seeds"]]

# NaN 값 확인 및 처리
print(train_x.isnull().sum())
print(train_y.isnull().sum()) #결측치 있음 

# NaN 값이 있는 행 삭제
train_y = train_y.dropna()  # train_x에서 NaN 값이 있는 행 제거
#train_y = train_y[train_x.index]  # train_y도 동일한 인덱스로 조정

############# 오늘 배운 코드로 체인 벨리데이션 해서 람다값 얻기 ##############
# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def mae(model):
    score = -cross_val_score(model, train_x, train_y, cv = kf,
                                     scoring = "neg_mean_absolute_error").mean()
    return score

lasso = Lasso(alpha=0.01) # 코드 잘 만들었는지 테스트
mae(lasso) # 테스트

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 1000, 10)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)
#0 

### 이제 나온 알파값으로 예상 해보자.
model = Lasso(alpha = 0)
model.fit(train_x, train_y)
model.coef_
model.intercept_

test_y_pred = model.predict(test_x)

sub_df["yield"] = test_y_pred #셈플에 가격 넣기
#sub_df.to_csv("./data/blueberry/sample_submission2.csv", index = False)
#lamda = 0, 366.44765
sub_df.to_csv("./data/blueberry/sample_submission3.csv", index = False)
#lamda = 0, 이상치를 평균값으로 대체, 366.44765