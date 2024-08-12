import pandas as pd 
import numpy as np
from scipy.stats import norm


x = np.array([79.1, 68.8, 62.0, 74.4, 71.0, 60.6, 98.5, 86.4, 73.0, 40.8, 61.2, 68.7, 61.6, 67.7, 61.7, 66.8])

#평균
x.mean()

#n(자료개수)
len(x)

#시그마



z_005 = norm.ppf(0.95, loc = 0, scale = 1) 
z_005

#신뢰구간 
x.mean() + z_005 * 6 / np.sqrt(16) #오른쪽
x.mean() - z_005 * 6 / np.sqrt(16) #왼쪽 


#Q. X ~ N(3, 5^2) 
#데이터로부터 E[X^2] 구하기 
x = norm.rvs(loc=3, scale=5, size = 10000)


np.mean(x**2) #n으로 나눈 것 

#E[(X-X^2)/(2X)] = ? => 데이터 많이 뽑고, 똑같이 식대로 만들면 된다.

np.mean((x-x**2)/(2*x)) #대략 -0.996이 나온다. 


#표본 10만개를 추출해서 s^2를 구해보세요. (20240729 시드 사용)
np.random.seed(20240729)
x = norm.rvs(loc=3, scale=5, size = 100000)

x_bar = x.mean()
s_2 = sum((x - x_bar)**2) / (100000 - 1)
s_2

np.var(x) 

np.var(x, ddof = 1) #n-1로 나눈 값 (표본 분산)


#n-1 vs. n
x = norm.rvs(loc=3, scale=5, size = 10)
np.var(x)
np.var(x, ddof=1) #n-1로 나눈 값 

#검정색 선, 파란색 선 추정 공부 
#y = 2x+3 그래프 그리기 
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# x 값의 범위 설정
x = np.linspace(0, 100, 400)  # -10부터 10까지 400개의 점 생성

# y 값 계산
y = 2 * x + 3  

np.random.seed(20240805)
obs_x = np.random.choice(np.arange(100), 20) #x의 값 랜덤하게 뽑기 #xi
epsilon_i = norm.rvs(loc = 0, scale = 20, size =20) 
obs_y = 2 * obs_x+3 + epsilon_i #yi 

# 그래프 그리기
plt.plot(x, y, label='y = 2x + 3', color='black')
plt.scatter(obs_x, obs_y, color = 'blue', s=3)
# plt.show()
# plt.clf()


###0805 5교시 
#obs 이용해서 회귀분석 a, b 추정 
import pandas as pd 
df = pd.DataFrame({"x": obs_x,
                   "y": obs_y})
df
#이 코드는 의미가 없다. 

from sklearn.linear_model import LinearRegression

# 선형 회귀 모델 생성
model = LinearRegression()

obs_x = obs_x.reshape(-1, 1)
# 모델 학습
model.fit(obs_x, obs_y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_[0]      # 기울기 a hat # array로 나와서 numpy로 나오게 하려고. 
model.intercept_ # 절편 b hat 

# x 값의 범위 설정
x = np.linspace(0, 100, 400)  # -10부터 10까지 400개의 점 생성
y = model.coef_[0] * x + model.intercept_

#빨간색 회귀직선 그리기 
plt.xlim([0, 100])
plt.ylim([0, 300])

plt.plot(x, y, color='red') #회귀직선
plt.show()
plt.clf()

model
summary(model)

#!pip install statsmodels 
import statsmodels.api as sm

obs_x = sm.add_constant(obs_x) 
model = sm.OLS(obs_y, obs_x).fit()
print(model.summary())


1- norm.cdf(18, loc=10, scale=1.96)

(18-10)/1.96
norm.cdf(4.08, loc = 0, scale = )


###0805 8교시
##p57

x = [15.078,15.752,15.549,15.56,16.098,13.277,15.462,16.116,15.214,16.93,14.118,14.927,
 15.382, 16.709, 16.804]
 
#표준편차를 모를 때 
np.mean(x)
np.std(x)



