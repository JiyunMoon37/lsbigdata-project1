import numpy as np
import matplotlib.pyplot as plt

#예제 넘파이 배열 생성
data = np.random.rand(10)

#히스토그램 그리기
plt.clf()
plt.hist(data, bins = 4, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
#plt.xlim(0, 1)  # x축 범위를 0에서 1 사이로 설정
plt.grid(False)
plt.show()

#실습
#1. 0~1 사이 숫자 5개 발생
#2. 표본 평균 계산하기
#3. 1,2 단계 10000번 반복한 결과를 벡터로 만들기
#4. 히스토그램 만들기

import numpy as np
import matplotlib.pyplot as plt

def Z(10000):
    return np.random.rand(5)

#0에서 1사이의 균등 분포에서 난수 5개 발생한다. 
#data = np.random.rand(5)
#data

#표본 평균 계산하기
mean_data = Z.mean()

plt.hist(Z, bins = 30, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


#0에서 1사이의 균등 분포에서 10000번 반복하여 난수 5개 발생한다. -> 50000
np.random.rand(50000).mean() 


# x = np.random.rand(50000).\
#     reshape(-1, 5).\
#     mean(axis = 1) 
x = np.random.rand(10000, 5).mean(axis = 1) 
plt.hist(x, bins = 30, alpha = 0.7, color = 'blue')
plt.title('Histogram of Numpy Vector')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(False)
plt.show()
plt.clf()

#0723 4교시 
import numpy as np

np.arange(33).sum()/33

np.unique((np.arange(33) - 16)**2)

sum(np.unique((np.arange(33) - 16)**2) * (2/33))


x = np.arange(33)
sym(x)/33
sum((x-16) *1/33)
(x-16)**2

np.unique((x-16)**2)*(2/33)
sum(np.unique((x-16)**2)*(2/33))

sum(x**2 * (1/33))

#Var(X) = E[X^2] - (E[X])^2
sum(x**2 * (1/33)) - 16**2

#EX
x = np.arange(4)
x
pro_x = np.array([1/6, 2/6, 2/6, 1/6])
pro_x

#기대값
Ex = sum(x * pro_x) 
Exx = sum(x**2 * pro_x)

#분산
Exx - Ex**2 
sum((x - Ex)**2 * pro_x) 

#0723_6교시
#연습문제1
x = np.arange(99)
x

#1~50까지 벡터 
x_1_50_1 = np.concatenate((np.arange(1, 51), np.arange(49, 0 ,-1))) 
pro_x = x_1_50_1/2500
pro_x

Ex = sum(x * pro_x)
Exx = sum(x**2 * pro_x)

#분산
Exx - Ex**2
sum((x-Ex)**2 ** pro_x)

#연습문제2 x = 0, 2, 4, 6
x = np.arange(4)*2
x

pro_x = np.array([1/6, 2/6, 2/6, 1/6])
pro_x

#기대값
Ex = sum(x * pro_x) 
Exx = sum(x**2 * pro_x)

#분산
Exx - Ex**2 
sum((x - Ex)**2 * pro_x) 

np.sqrt(9.52**2 / 10)

81 / 25
np.sqrt(3.24**0.5)


#이항분포 X ~ P(X = k | n, p)
#n : 베르누이 확률변수 더한 갯수
#p : 1이 나올 확률
#binom, pmf(k , n, p)
#pip install scipy를 터미널에 쳐준다. 
from scipy.stats import bernoulli
from scipy.stats import binom 
binom.pmf(0, n=2, p=0.3)
binom.pmf(1, n=2, p=0.3)
binom.pmf(2, n=2, p=0.3)

binom.pmf(0, n=30, p=0.3)


#X ~ B(n. p)
#list comp.
result =[binom.pmf(x, n=30, p=0.3) for x in range(31)]
result


#numpy
import numpy as np

import math
(math.factorial(54) / (math.factorial(26)*math.factorial(28)))

math.comb(54, 26)

n = 54
factorial_54 = np.cumprod(range(1, n+1))[-1]
factorial_54

n1 = 28
factorial_28 = np.cumprod(range(1, n1+1))[-1]
factorial_28

n2 = 26
factorial_26 = np.cumprod(range(1, n2+1))[-1]
factorial_26

factorial_54 / (factorial_28*factorial_26)

fac_54 = np.cumprod(np.arange(1, 55))[-1]

import math
#log(a*b) = log(a) + log(b)

math.log(10*100) = math.log(10) + math.log(100)

log(1*2*3*4) = log(1) + log(2) +log(3) + log(4) 
np.log(24)
sum(np.log(np.arange(1, 5)))

math.log(math.factorial(54))


#log(54!)
a = math.log(math.factorial(54))

#log(26!)
b = math.log(math.factorial(26))

#log(28!)
c = math.log(math.factorial(28))

d = a - (b + c)
np.exp(d)
#==============
math.comb(2, 0) * 0.3**0  * (1-0.3)**2 
math.comb(2, 1) * 0.3**1  * (1-0.3)**1
math.comb(2, 2) * 0.3**2  * (1-0.3)**0

#pmf는 질량
binom.pmf(0, 2, 0.3) 
binom.pmf(1, 2, 0.3)
binom.pmf(2, 2, 0.3)


#예제
#X~B(n=10, p=0.36)
#P(X=4)
binom.pmf(4, n=10, p=0.36) 

#예제
#X~B(n=10, p=0.36)
#P(X<=4)
binom.pmf(np.arange(5), n=10, p=0.36).sum()


#X~B(n=10, p=0.36)
#P(2<X<=8)
np.arange(3,9)
binom.pmf(np.arange(3,9), n=10, p=0.36).sum()

#X~B(30, 0.2)
#확률변수 X가 4보다 작거나 25보다 크거나 같을 확률을 구하시오. - 수식으로 먼저 적어보기 
#P(X<4 or X>= 25)

#1번 풀이
np.arange(4)
np.arange(25, 31)
#1
a = binom.pmf(np.arange(4), n=30, p=0.2).sum()
#2
b = binom.pmf(np.arange(25, 31), n=30, p=0.2).sum() 
#3
a+b

np.arange(4,26)
#4
binom.pmf(np.arange(4,26), n=30, p=0.2).sum()
1- binom.pmf(np.arange(4,26), n=30, p=0.2).sum()

#rvs함수 (random variates sample)
#표본 추출 함수
#X1 ~ Bernulli(p=0.3)
bernoulli.rvs(p=0.3)
#X2 ~ Bernulli(p=0.3)
#X ~ B(n=2, p=0.3)
bernoulli.rvs(0.3) + bernoulli.rvs(0.3) 
binom.rvs(2, 0.3, 1) 
binom.rvs(n=2, p=0.3, size=1)
binom.pmf(0, n =2, p=0.3)
binom.pmf(1, n =2, p=0.3)
binom.pmf(2, n =2, p=0.3)

#X~B(30, 0.26)
#표본 30개를 뽑아보세요!
binom.rvs(n = 30, p = 0.26, size = 30) 
#0에서 30까지의 숫자가 올 수 있고, 각 확률은 0.26, 난수 30개

#X~B(30, 0.26)
#기댓값 30*0.26 = 7.8
#표본 30개를 뽑아보세요!

import seaborn as sns
prob_x = binom.pmf(np.arange(31), n=30, p=0.26) 

sns.barplot(prob_x)
import matplotlib.pyplot as plt
plt.show()
plt.clf() 

#교재 p207
import pandas as pd 
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26) 

df = pd.DataFrame({"x" :x, "prob": prob_x})
df

sns.barplot(data = df, x = "x", y = "prob")
plt.show()

#gpt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom

# 이항 분포에서 난수 생성
n = 30
p = 0.26
size = 30
samples = binom.rvs(n=n, p=p, size=size)

# 히스토그램 그리기
plt.figure(figsize=(8, 6))
plt.hist(samples, bins=np.arange(0, n+1, 1), edgecolor='black', facecolor='lightblue')
plt.xlabel('성공 횟수')
plt.ylabel('빈도')
plt.title(f'Binomial Distribution (n={n}, p={p:.2f})')
plt.grid(True)
plt.show()

#4교시 
#cdf :culmulative dist. function
#(누적확률분포 함수)
#F_X(x) = P(X <= x)
binom.cdf(4, n=30, p =0.26) #cdf는 항상 작거나 같다 

#P(4<x<=18) = ?
binom.cdf(18, n=30, p=0.26) - binom.cdf(4, n=30, p=0.26) 

#P(13<x<20) = ?

binom.cdf(19, n=30, p=0.26) - binom.cdf(13, n=30, p=0.26) 

#
import numpy as np
import seaborn as sns

x_1 = binom.rvs(n=30, p=0.26, size=3)
x_1
x = np.arange(31)
prob_x = binom.pmf(x, n=30, p=0.26)
sns.barplot(prob_x, color = "blue")
plt.show()

#add a point at (2,0)
plt.scatter(x_1, np.repeat(0.002,3), 
            color='red', zorder=100, s = 10)
#zorder = 5 : 점의 z순서를 5로 설정
#s=5 : 점의 크기를 5로 설정 

#기댓값 표현
plt.axvline(x=7.8, color = 'green',
            linestyle = '--', linewidth=2)

plt.show()
plt.clf()

binom.ppf(0.5,n=30, p = 0.26 )
binom.ppf(0.7,n=30, p = 0.26 ) #첫번째 입력 값이 0~1사이 숫자가 들어가야함.
binom.cdf(8, n = 30, p = 0.26)
binom.cdf(9, n = 30, p = 0.26)

#정규분포
import math
1 / np.sqrt(2*math.pi)
from scipy.stats import norm

norm.pdf(0, loc = 0, scale = 1)

#문제 loc = 3, scale = 4, x = 5
norm.pdf(5, loc = 3, scale = 4)

#-3부터 3까지 sequence 발생, 그 값에 해당하는 pdf값 가져오기 
#값 5개 가져오기 
plt.clf()
k = np.linspace(-5, 5, 100) 
y = norm.pdf(k, loc = 0, scale = 1)

plt.scatter(k, y, color='red')
plt.show()

#6교시
k = np.linspace(-5, 5, 100) 
y = norm.pdf(k, loc = 0, scale = 1)

plt.plot(k, y, color = "black")
plt.show()
plt.clf()

#mu (loc): 분포의 중심 결정하는 모수 
k = np.linspace(-5, 5, 100) 
y = norm.pdf(k, loc = 3, scale = 1)

plt.plot(k, y, color = "black")
plt.show()
plt.clf()

k = np.linspace(-5, 5, 100) 
y = norm.pdf(k, loc = -3, scale = 1)

plt.plot(k, y, color = "black")
plt.show()
plt.clf()

k = np.linspace(-5, 5, 100) 
y = norm.pdf(k, loc =0, scale = 1)
y2 = norm.pdf(k, loc =0, scale = 2)
y3 = norm.pdf(k, loc = 0, scale =0.5)

plt.plot(k, y, color = "black")
plt.plot(k, y2, color = "red")
plt.plot(k, y3, color = "blue")

plt.show()
plt.clf()

표준편차가 작다 = 평균 근처에 값이 퍼져있는 정도가 작다. 
정규분포는 종모양, 평균은 모양을 결정하고, 시그마는 퍼짐 정도를 결정한다. 

norm.cdf(0, loc=0, scale=1)
norm.cdf(100, loc=0, scale=1)

#P(-2 < X < 0.54)
norm.cdf(0.54, loc=0, scale=1) - norm.cdf(-2, loc=0, scale=1)

#P(X<1 or X>3)
#1
cdf(1,0,1)
#2
1-cdf(3,0,1)

a = norm.cdf(1, loc=0, scale=1)
b = norm.cdf(3, loc=0, scale=1)

a+ (1-b)

#X~N(3, 5^2) 5^2는 분산, 표준편차는 5
#P(3<X<5)=?

a = norm.cdf(3,loc = 3, scale = 5)
b = norm.cdf(5,loc = 3, scale = 5)

b-a 

#X~N(3, 5^2) 5^2는 분산, 표준편차는 5
#P(3<X<5)=?
#위 확률변수에서 표본 1000개 뽑기 
x = norm.rvs(loc = 3, scale = 5, size=1000) 
x

#x > 3 & x < 5
(x > 3) & (x < 5)
sum((x > 3) & (x < 5)) / 1000

#평균 : 0, 표준편차 : 1
#표본 1000개 뽑아서 0보다 작은 비율 확인

y = norm.rvs(loc = 0, scale = 1, size=1000) 
y
#o보다 작은 비율 확인
y < 0
sum(y<0) / 1000

#평균 : 0, 표준편차 :1
#표본 1000개 뽑아서 0보다 작은 비율 확인
import numpy as np
from scipy.stats import norm
import seaborn as sns
import matplotlib.pyplot as plt

x = norm.rvs(loc = 0, scale = 1, size = 1000)
np.mean(x < 0)
sum(x < 0)/1000

x = norm.rvs(loc = 3, scale = 2, size = 1000)
x

sns.histplot(x)
plt.show()
plt.clf() 

====
x = norm.rvs(loc = 3, scale = 2, size = 1000)
x

#sns.histplot(x)
sns.histplot(x, stat = "density")

#plot he normal distribution PDF
xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc = 3, scale=2)
plt.plot(x_values, pdf_values, color = 'red', linewidth = 2)

plt.show()
plt.clf() 
=====
x = norm.rvs(loc=3, scale=2, size =1000)
x
sns.histplot(x, stat = "density")

#plot the normal distribution PDF
xmin, xmax = (x.min(), x.max())
x_values = np.linspace(xmin, xmax, 100)
pdf_values = norm.pdf(x_values, loc =3, scale =2)
plt.plot(x_values, pdf_values, color = "red", linewidth = 2)

plt.show()
plt.clf()

score = norm.ppf(0.95, loc=30, scale=2)
score = norm.f(0.95, loc=30, scale=2)


print(score)
















































