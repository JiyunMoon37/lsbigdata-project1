#soft copy
a = [1, 2, 3]
a

b = a
b

a[1] = 4
a

b

#deep copy
a = [1, 2, 3]
a

b = a[:]
b = a.copy()

a[1] = 4
a
b 

id(a)
id(b)

#수학함수
import math
x = 4
math.sqrt(x) 

#지수 계산 예제
exp_val = math.exp(5)
print("e^5의 값은:", exp_val)

#로그 계산 예제
log_val = math.log(10, 10)
print("10의 밑 10 로그 값은:", log_val)

#수학함수
import math
x = 4
math.sqrt(x) 

exp_val = math.exp(5)
exp_val

def my_normal_pdf(x, mu, sigma):
  part_1 = (sigma * math.sqrt(2 * math.pi))**(-1) 
  part_2 = math.exp((-(x-mu)**2) / (2*sigma**2))
  return part_1 * part_2
  
my_normal_pdf(3, 3, 1)

def normal_pdf(x, mu, sigma) : 
  sqrt_two_pi = math.sqrt(2 * math.pi)
  factor = 1 / (sigma * sqrt_two_pi)
  return factor * math.exp( -0.5 * ((x-mu) / sigma) ** 2)



def my_f(x, y, z) : 
  return (x**2 + math.sqrt(y) + math.sin(z)) * math.exp(x) 

my_f(2, 9, math.pi/2)


def my_g(x) :
  return math.cos(x) + math.sin(x) * math.exp(x) 

my_g(math.pi)


#강의록4 
#!pip install numpy
#import pandas as pd 
import numpy as np
        

 # 벡터 생성하기 예제
a = np.array([1, 2, 3, 4, 5])  # 숫자형 벡터 생성
b = np.array(["apple", "banana", "orange"])  # 문자형 벡터 생성
c = np.array([True, False, True, True])  # 논리형 벡터 생성
print("Numeric Vector:", a)
print("String Vector:", b)
print("Boolean Vector:", c)
 
type(a) #얘만이 쓸 수 있는 함수로 변경 

a[3] 
a[2:]
a[1:4]

b = np.empty(3)
b

b[0]=1
b[1]=4
b[2]=10
b
b[2]

vec1 = np.array([1,2,3,4,5])
vec1 = np.arange(100)
vec1 = np.arange(1,100.5, 0.5)
vec1

l_space1 = np.linspace(0, 1, 5)

linear_space2 = np.linspace(0, 1, 5, endpoint = False)
linear_space2
?np.linspace

#-100부터 0까지
vec2 = np.arange(0, -101, -1)
vec2
vec3 = -np.arange(0, 101)
vec3

vec1 = np.arange(5)
np.repeat(vec1, 5)

#repeat vs. tile
vec1 = np.arange(5)
np.repeat(vec1,3)
np.tile(vec1, 3)

vec1 * 2 
vec1 +vec1
vec1 = np.array([1, 2, 3, 4])


max(vec1)
sum(vec1)

#35672 이하 홀수들의 합은? 
vec4 = np.arange(1, 35673, 2)
vec4
sum(vec4)

vec4 = np.arange(1, 35673, 2).sum()
vec4

sum(np.arange(1, 35673, 2))
x = np.arange(1, 35673, 2)
x.sum()

len(x)
x.shape

[[1, 2, 3], [4, 5, 6]]

b = np.array([[1, 2, 3], [4, 5, 6]])
b

length = len(b)
shape = b.shape
size = b.size

a = np.array([1,2])
b = np.array([1,2,3,4])

a
b

a+b
np.tile(a,2) + b
np.repeat(a,2)+b
b == 3

#35672보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는? 
x = np.arange(1, 35672)
(x%7)==3
sum(x%7 == 3)

#10보다 작은 수 중에서 7로 나눠서 나머지가 3인 숫자들의 갯수는? 
x = np.arange(1, 10)
(x%7)==3
sum((x%7)==3)
sum(x%7 == 3)


a = np.array([1.0, 2.0, 3.0])
 b = 2.0
 a * b
 
 a.shape
b.shape 


import numpy as np
 # 2차원 배열 생성
matrix = np.array([[ 0.0,  0.0,  0.0],
                   [10.0, 10.0, 10.0],
                   [20.0, 20.0, 20.0],
                   [30.0, 30.0, 30.0]])

 matrix.shape
 # 1차원 배열 생성
vector = np.array([1.0, 2.0, 3.0])
 vector.shape
 # 브로드캐스팅을 이용한 배열 덧셈
result = matrix + vector
 print("브로드캐스팅 결과:\n", result)
 
# 세로 벡터 생성
vector = np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
vector
vector.shape

result = matrix + vector
print("브로드캐스팅 결과:\n", result)
 
np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1)
