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












