import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import uniform
from sklearn.linear_model import LinearRegression

# 20차 모델 성능을 알아보자
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

import pandas as pd
df = pd.DataFrame({
    "y" : y,
    "x" : x
})
df

train_df = df.loc[:19]
train_df

for i in range(2, 21):
    train_df[f"x{i}"] = train_df["x"] ** i
    
# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
train_y = train_df["y"]

valid_df = df.loc[20:]
valid_df

for i in range(2, 21):
    valid_df[f"x{i}"] = valid_df["x"] ** i

# 'x' 열을 포함하여 'x2'부터 'x20'까지 선택.
valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
valid_x
valid_y = valid_df["y"]
valid_y

from sklearn.linear_model import Lasso

# 결과 받기 위한 벡터 만들기
val_result=np.repeat(0.0, 100)
tr_result=np.repeat(0.0, 100)

for i in np.arange(0, 100):
    model= Lasso(alpha=i*0.01)
    model.fit(train_x, train_y)

    # 모델 성능
    y_hat_train = model.predict(train_x)
    y_hat_val = model.predict(valid_x)

    perf_train=sum((train_df["y"] - y_hat_train)**2)
    perf_val=sum((valid_df["y"] - y_hat_val)**2)
    tr_result[i]=perf_train
    val_result[i]=perf_val

tr_result
val_result

import seaborn as sns

df = pd.DataFrame({
    'lambda': np.arange(0, 1, 0.01), 
    'tr': tr_result,
    'val': val_result
})

# seaborn을 사용하여 산점도 그리기
sns.scatterplot(data=df, x='lambda', y='tr')
sns.scatterplot(data=df, x='lambda', y='val', color='red')
plt.xlim(0, 0.4)

val_result[0]
val_result[1]
np.min(val_result)

# alpha를 0.03로 선택!
np.argmin(val_result)
np.arange(0, 1, 0.01)[np.argmin(val_result)]

model = Lasso(alpha=0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_
#model.predict(test_x)

sorted_train = train_x.sort_values("x")
reg_line = model.predict(sorted_train)

plt.plot(valid_train["x"], reg_line, color = "red")
plt.scatter(valid_df["x"], valid_df["y"], )

#0827 3교시
##실습 
#lamda=0.03인 라쏘 
model = Lasso(alpha = 0.03)
model.fit(train_x, train_y)
model.coef_
model.intercept_

#그리기
k = np.arange(-4, 4, 0.01)

k_df = pd.DataFrame({
    "x" : k
})
k_df

for i in range(2, 21):
    k_df[f"x{i}"] = k_df["x"] ** i

legline = model.predict(k_df)

plt.plot(k, legline, color = "red")
plt.scatter(valid_df["x"], valid_df["y"], color = "blue")


#0827 4교시
##train셋을 5개로 쪼개어 valid set을 5개 만들기
##각 세트에 대한 성능을 각 lamda 값에 대응하여 구하기
##성능 평가 지표 5개를 평균내어 오른쪽 그래프 다시 그리기 
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

myindex = np.random.choice(np.arange(30), 30, replace = False)


##gpt
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform

# 데이터 생성
np.random.seed(2024)
x = uniform.rvs(size=30, loc=-4, scale=8)
y = np.sin(x) + norm.rvs(size=30, loc=0, scale=0.3)

df = pd.DataFrame({"y": y, "x": x})

# 다항 피처 생성
for i in range(2, 21):
    df[f"x{i}"] = df["x"] ** i

# 데이터 나누기
n = len(df)
fold_size = n // 5
val_results = []

alpha_values = np.arange(0, 1, 0.01)

# 각 alpha에 대해 성능 평가
for alpha in alpha_values:
    fold_results = []
    
    for fold in range(5):
        # 학습 및 검증 데이터 인덱스 설정
        valid_index = list(range(fold * fold_size, (fold + 1) * fold_size))
        train_index = list(set(range(n)) - set(valid_index))
        
        train_df = df.iloc[train_index]
        valid_df = df.iloc[valid_index]
        
        train_x = train_df[["x"] + [f"x{i}" for i in range(2, 21)]]
        train_y = train_df["y"]
        
        valid_x = valid_df[["x"] + [f"x{i}" for i in range(2, 21)]]
        valid_y = valid_df["y"]
        
        # Lasso 모델 학습
        model = Lasso(alpha=alpha)
        model.fit(train_x, train_y)
        
        # 예측 및 성능 평가
        y_hat_val = model.predict(valid_x)
        perf_val = sum((valid_y - y_hat_val) ** 2)
        fold_results.append(perf_val)
    
    # 각 alpha에 대한 평균 성능 저장
    val_results.append(np.mean(fold_results))

# 결과 DataFrame 생성
result_df = pd.DataFrame({
    'lambda': alpha_values,
    'val': val_results
})

# 그래프 그리기
plt.figure(figsize=(10, 6))
plt.plot(result_df['lambda'], result_df['val'], marker='o', color='red')
plt.title('Lasso Regression Performance by Alpha Value (Manual Fold)')
plt.xlabel('Lambda Values')
plt.ylabel('Mean Squared Error (MSE)')
plt.xlim(0, 0.4)
plt.grid()
plt.show()











#gpt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

# 데이터 생성
k = np.arange(-4, 4, 0.01)
k_df = pd.DataFrame({"x": k})

# 다항식 피처 생성
for i in range(2, 21):
    k_df[f"x{i}"] = k_df["x"] ** i

# 예시 데이터: valid_df는 실제 데이터로 교체 필요
valid_df = pd.DataFrame({"x": np.random.uniform(-4, 4, 100), "y": np.random.uniform(-10, 10, 100)})

# 모델 정의
model = LinearRegression()

# 교차 검증
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_index, test_index in kf.split(valid_df):
    train_df = valid_df.iloc[train_index]
    test_df = valid_df.iloc[test_index]
    
    # 피처와 타겟 변수 정의
    X_train = train_df[['x']].copy()
    for i in range(2, 21):
        X_train[f"x{i}"] = X_train["x"] ** i
        
    y_train = train_df["y"]
    
    # 모델 학습
    model.fit(X_train, y_train)
    
    # 예측
    X_test = test_df[['x']].copy()
    for i in range(2, 21):
        X_test[f"x{i}"] = X_test["x"] ** i
        
    y_pred = model.predict(X_test)
    
    # 성능 평가
    score = model.score(X_test, test_df["y"])
    scores.append(score)

# 평균 점수 출력
print(f'교차 검증 평균 점수: {np.mean(scores)}')

# 최종 모델 예측 및 시각화
legline = model.predict(k_df)

plt.figure(figsize=(10, 6))
plt.axvline(x=0.03, color='green', linestyle='--', label='lambda=0.03')
plt.plot(k, legline, color="red", label="모델 예측")
plt.scatter(valid_df["x"], valid_df["y"], color="blue", label="실제 데이터")
plt.xlim(0, 0.4)
plt.legend()
plt.show()
