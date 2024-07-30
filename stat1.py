#균일 확률변수 만들기
import numpy as np

np.random.rand(1)

def X(i):
    return np.random.rand(i)
X()

X(1)
X(2)
X(3)

#베르누이 확률변수 모수 : p 만들어보세요!
num = 3 
p = 0.5 
def Y(num, p):
    x = np.random.rand(num)
    return np.where(x < p, 1, 0) #p보다 작을 때 1, p보다 클 때 0
Y(num = 5, p = 0.5)
Y(num = 100, p = 0.5)
sum(Y(num = 100, p = 0.5))/100
Y(num = 100, p = 0.5).mean()
Y(num = 100000, p = 0.5).mean()


#새로운 확률변수
#가질 수 있는 값 : 0, 1, 2
#20%, 50%, 30%
import numpy as np

p = [0.2, 0.5, 0.3] #0은 20%, 1은 50%, 2는 30%를 의미 
def Y(1, p):
    x = np.random.rand(1)
    return np.where((x < p[0]), 0, 
    np.where((x<p[0] + p[1]), 1, 2))[0]
Y(1, p)

def Z(p1, p2, p3) :
    x = np.random.rand(3)
    return np.where(x[0] < p1, 0, (x[1] <p2, 1, 2))

Z(0.2, 0.5, 0.3)


def Z():
    x = np.random.rand(1)
    return np.where((x < 0.2), 0, np.where((x < 0.7), 1,2))
Z()

#p이용

def Z(p):
    x = np.random.rand(1)
    p_cumsum = p.cumsum()
    return np.where((x < p_cumsum[0]), 0, np.where(x<p_cumsum[1], 1, 2))

p = np.array([0.2, 0.5, 0.3]) #0은 20%, 1은 50%, 2는 30%를 의미 
Z(p)

#E[X]
import numpy as np
sum(np.arange(4) * np.array([1, 2, 2, 1]) / 6)

import numpy as np
import matplotlib.pyplot as plt

# 모수 p 값 설정
p = 0.6

# 확률변수 x의 값 범위 설정
x = [0, 1]

# 확률질량함수 계산
pmf = [1-p, p]

# 그래프 그리기
plt.figure(figsize=(8, 6))
plt.bar(x, pmf)
plt.xlabel('x')
plt.ylabel('P(x)')
plt.title(f'Bernoulli Distribution (p={p})')
plt.grid()
plt.show()

def Y(num) : 
    x = np.random.rand(num)
    return np.where( x <p , 1, 0)

Y(num=30)
sum(np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1]))

def Z():
    x = np.random.rand(1)
    p_cumsum = p.cumsum()
    return np.where((x < p_cumsum[0]), 0, np.where(x<p_cumsum[1], 1, 2))

p = np.array([0.2, 0.5, 0.3]) #0은 20%, 1은 50%, 2는 30%를 의미 
Z()


sum(np.arange(4) * np.array([1, 2, 2, 1]) / 6)






























