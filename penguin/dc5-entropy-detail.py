import pandas as pd
import numpy as np
from palmerpenguins import load_penguins

penguins = load_penguins()
penguins.head()

##0906 1교시 
#펭귄 분류 문제
#y : 펭귄의 종류
#x1 : bill_length_mm (부리길이)
#x2 : bill_depth_mm (부리 깊이)

df=penguins.dropna()
df=df[["bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={
    'species' : 'y',
    'bill_length_mm': 'x1',
    'bill_depth_mm': 'x2'})
df

#x1, x2 산점도를 그리되, 점 색깔은 펭귄 종별로 다르게 그리기!
import matplotlib.pyplot as plt
import seaborn as sns

sns.scatterplot(data=df, x = "x1", y = "x2", hue ='y')
plt.axvline(x=45)

#Q. 나누기 전 현재의 엔트로피?
#Q. 45로 나눴을때, 엔트로피 평균은 얼마인가요?
#입력값이 벡터 -> 엔트로피!
p_i=df['y'].value_counts() / len(df['y'])
entropy_curr=-sum(p_i * np.log2(p_i))

# x=45 기준으로 나눈 후, 평균 엔트로피 구하기!
# 10분!

# x1=45 기준으로 나눈 후, 평균 엔트로피 구하기!
# x1=45 기준으로 나눴을때, 데이터포인트가 몇개 씩 나뉘나요?
n1=df.query("x1 < 45").shape[0]  # 1번 그룹
n2=df.query("x1 >= 45").shape[0] # 2번 그룹

# 1번 그룹은 어떤 종류로 예측하나요?
# 2번 그룹은 어떤 종류로 예측하나요?
y_hat1=df.query("x1 < 45")['y'].mode()
y_hat2=df.query("x1 >= 45")['y'].mode()

# 각 그룹 엔트로피는 얼마 인가요?
p_1=df.query("x1 < 45")['y'].value_counts() / len(df.query("x1 < 45")['y'])
entropy1=-sum(p_1 * np.log2(p_1))

p_2=df.query("x1 >= 45")['y'].value_counts() / len(df.query("x1 >= 45")['y'])
entropy2=-sum(p_2 * np.log2(p_2))

entropy_x145=(n1 * entropy1 + n2 * entropy2)/(n1 + n2)
entropy_x145

##0906 2교시
#엔트로피를 만들고 
#x1 기준으로 최적 기준값 찾기
#기준값 x를 넣으면 엔트로피 값이 나오는 함수는?
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

penguins = load_penguins()
df = penguins.dropna()
df = df[["species", "bill_length_mm", "bill_depth_mm"]]
df = df.rename(columns={
    'species': 'y',
    'bill_length_mm': 'x1',
    'bill_depth_mm': 'x2'})

def my_entropy(x):
    n1 = df.query(f"x1 < {x}").shape[0]
    n2 = df.query(f"x1 >= {x}").shape[0]
    
    if n1 == 0 or n2 == 0:
        return np.inf 

    p_1 = df.query(f"x1 < {x}")['y'].value_counts() / n1
    entropy1 = -sum(p_1 * np.log2(p_1)) if n1 > 0 else 0

    p_2 = df.query(f"x1 >= {x}")['y'].value_counts() / n2
    entropy2 = -sum(p_2 * np.log2(p_2)) if n2 > 0 else 0

    return (n1 * entropy1 + n2 * entropy2) / (n1 + n2)

result = minimize_scalar(my_entropy, bounds=(df["x1"].min(), df["x1"].max()), method='bounded')
optimal_threshold = result.x
optimal_entropy = result.fun

print(f"최적 기준값: {optimal_threshold}")
print(f"해당 기준값에서의 엔트로피: {optimal_entropy}")

#승학오빠 코드
def my_entropy(x):
    n1=df.query(f"x1 < {x}").shape[0]  # 1번 그룹
    n2=df.query(f"x1 >= {x}").shape[0] # 2번 그룹   
    p_1=df.query(f"x1 < {x}")['y'].value_counts() / len(df.query(f"x1 < {x}")['y'])
    entropy1=-sum(p_1 * np.log2(p_1))
    p_2=df.query(f"x1 >= {x}")['y'].value_counts() / len(df.query(f"x1 >= {x}")['y'])
    entropy2=-sum(p_2 * np.log2(p_2))
    return float((entropy1 * n1 + entropy2 * n2)/(n1+n2))

my_entropy(45)

result = []
x1_values = np.arange(df["x1"].min(),df["x1"].max()+1,0.01)
for x in x1_values:
    result.append(entropy(x))
result
x1_values[np.argmin(result)]



#용규오빠
import pandas as pd
import numpy as np
from palmerpenguins import load_penguins
import matplotlib.pyplot as plt
penguins = load_penguins()
penguins.head()

# 펭귄 분류 문제
# y: 펭귄의 종류
# x1: bill_length_mm (부리 길이)
# x2: bill_depth_mm (부리 깊이)

df=penguins.dropna()
df=df[["species", "bill_length_mm", "bill_depth_mm"]]
df=df.rename(columns={
    'species': 'y',
    'bill_length_mm': 'x1',
    'bill_depth_mm': 'x2'})
df


# x1, x2 산점도를 그리되, 점 색깔은 펭귄 종별 다르게 그리기!
import seaborn as sns

sns.scatterplot(data=df, x="x1", y="x2", hue='y')
plt.axvline(x=45)

# Q. 나누기 전 현재의 엔트로피?
# Q. 45로 나눴을때, 엔트로피 평균은 얼마인가요?
# 입력값이 벡터 -> 엔트로피!
p_i=df['y'].value_counts() / len(df['y'])
entropy_curr=-sum(p_i * np.log2(p_i))

# x1=45 기준으로 나눈 후, 평균 엔트로피 구하기!
# x1=45 기준으로 나눴을때, 데이터포인트가 몇개 씩 나뉘나요?
n1=df.query("x1 < 45").shape[0]  # 1번 그룹
n2=df.query("x1 >= 45").shape[0] # 2번 그룹

# 1번 그룹은 어떤 종류로 예측하나요?
# 2번 그룹은 어떤 종류로 예측하나요?
y_hat1=df.query("x1 < 45")['y'].mode()
y_hat2=df.query("x1 >= 45")['y'].mode()

#x1 기준으로 최적 기준값 찾기
# 기준값 x를 넣으면 엔트로피 값이 나오는 함수는?

def entropy(n,df):
    df_left=df.query(f"x1 < {n}")  # 1번 그룹
    df_right=df.query(f"x1 >= {n}") # 2번 그룹
    n_left=df_left.shape[0]  # 1번 그룹 n
    n_right=df_right.shape[0] # 2번 그룹 n
    p_left=df_left['y'].value_counts() / n_left
    entropy_left=-sum(p_left * np.log2(p_left))
    p_right=df_right['y'].value_counts() / n_right
    entropy_right=-sum(p_right * np.log2(p_right))
    return (n_left * entropy_left + n_right * entropy_right)/(n_left + n_right)

result = []
x1_values = np.arange(df["x1"].min(),df["x1"].max()+1,0.1)
for x in x1_values:
    result.append(entropy(x,df))
result
x1_values[np.argmin(result)]

plt.plot(x1_values, result)
plt.xlabel('x1')
plt.ylabel('entropy')
plt.show()







# 원래 MSE는?
np.mean((df["y"] - df["y"].mean())**2)
29.81

# x=15 기준으로 나눴을때, 데이터포인트가 몇개 씩 나뉘나요?
# 57, 276
n1=df.query("x < 15").shape[0]  # 1번 그룹
n2=df.query("x >= 15").shape[0] # 2번 그룹

# 1번 그룹은 얼마로 예측하나요?
# 2번 그룹은 얼마로 예측하나요?
y_hat1=df.query("x < 15").mean()[0]
y_hat2=df.query("x >= 15").mean()[0]

# 각 그룹 MSE는 얼마 인가요?
mse1=np.mean((df.query("x < 15")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 15")["y"] - y_hat2)**2)

# x=15 의 MSE 가중평균은?
# (mse1 + mse2)*0.5 가 아닌
(mse1* n1 + mse2 * n2)/(n1+n2)
29.23

29.81 - 29.23

# x = 20일때 MSE 가중평균은?
n1=df.query("x < 20").shape[0]  # 1번 그룹
n2=df.query("x >= 20").shape[0] # 2번 그룹
y_hat1=df.query("x < 20").mean()[0]
y_hat2=df.query("x >= 20").mean()[0]
mse1=np.mean((df.query("x < 20")["y"] - y_hat1)**2)
mse2=np.mean((df.query("x >= 20")["y"] - y_hat2)**2)
(mse1* n1 + mse2 * n2)/(n1+n2)
29.73

29.81-29.73

df=df.query("x < 16.41")

# 기준값 x를 넣으면 MSE값이 나오는 함수는?
def my_mse(x):
    n1=df.query(f"x < {x}").shape[0]  # 1번 그룹
    n2=df.query(f"x >= {x}").shape[0] # 2번 그룹
    y_hat1=df.query(f"x < {x}").mean()[0]
    y_hat2=df.query(f"x >= {x}").mean()[0]
    mse1=np.mean((df.query(f"x < {x}")["y"] - y_hat1)**2)
    mse2=np.mean((df.query(f"x >= {x}")["y"] - y_hat2)**2)
    return float((mse1* n1 + mse2 * n2)/(n1+n2))

my_mse(15)
my_mse(13.71)
my_mse(14.01)

df["x"].min()
df["x"].max()

# 13~22 사이 값 중 0.01 간격으로 MSE 계산을 해서
# minimize 사용해서 가장 작은 MSE가 나오는 x 찾아보세요!
x_values=np.arange(13.2, 16.4, 0.01)
nk=x_values.shape[0]
result=np.repeat(0.0, nk)
for i in range(nk):
    result[i]=my_mse(x_values[i])

result
x_values[np.argmin(result)]
# 14.01, 16.42, 19.4

# x, y 산점도를 그리고, 빨간 평행선 4개 그려주세요!
import matplotlib.pyplot as plt

df.plot(kind="scatter", x="x", y="y")
thresholds=[14.01, 16.42, 19.4]
df["group"]=np.digitize(df["x"], thresholds)
y_mean=df.groupby("group").mean()["y"]
k1=np.linspace(13, 14.01, 100)
k2=np.linspace(14.01, 16.42, 100)
k3=np.linspace(16.42, 19.4, 100)
k4=np.linspace(19.4, 22, 100)
plt.plot(k1, np.repeat(y_mean[0],100), color="red")
plt.plot(k2, np.repeat(y_mean[1],100), color="red")
plt.plot(k3, np.repeat(y_mean[2],100), color="red")
plt.plot(k4, np.repeat(y_mean[3],100), color="red")