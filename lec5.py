import numpy as np

np.random.seed(42)
a = np.random.randint(1, 21, 10)
<<<<<<< HEAD
a = np.random.choice(np.arange(1, 21), 10, False)
print(a)

a = np.random.choice(np.arange(1, 4), 100, True, np.array([2/5, 2/5, 1/5]))
a

=======
print(a)

>>>>>>> 82e1c4302480b7c0286914e65cc9c6708bfd4376
print(a[1])

a[2:5]
a[-1]
a[-2]
a[::]
a[::1]
a[::2]

#1에서부터 1000사이 3의 배수의 합은?
sum(np.arange(1,1001) % 3 == 0)
sum(np.arange(3, 1001, 3))

x = np.arange(1, 1001)
sum(x[2:1000:3])
sum(x[::3])

print(a[[0, 2, 4]])
a
print(np.delete(a,1))


a > 3
a[a>3]

print(b)

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
a[(a >2000) & (a < 5000)] #a[조건을 만족하는 논리형 벡터터]

a > 2000
a < 5000


#!pip install pydataset
import pydataset

df = pydataset.data('mtcars')
np_df = np.array(df['mpg'])

model_names = np.array(df.index)

#15이상 25 이하인 데이터 개수는? 
sum((np_df >= 15) & (np_df <= 25))

#평균 mpg보다 높은(이상) 자동차 대수는?
sum(np_df >= np.mean(np_df))

#15보다 작거나 22 이상인 데이터 개수는? 
sum((np_df < 15) | (np_df >= 22))

np.random.seed(2024)
a = np.random.randint(1, 10000, 5)
b = np.array(["A", "B", "C", "F", "W"])

#a[조건을 만족하는 논리형 벡터]
a[(a > 2000) & (a < 5000)]
b[(a > 2000) & (a < 5000)]

a[a > 3000] = 3000
a

#15이상 25 이하인 자동차 모델은? 
model_names[(np_df >= 15) & (np_df <= 25)]

#평균 mpg보다 높은(이상) 자동차 모델은?
model_names[np_df >= np.mean(np_df)]

#평균 mpg보다 낮은(미만) 자동차 모델은?
model_names[np_df < np.mean(np_df)]

np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a

#처음으로 22000보다 큰 숫자가 나왔을 때, 숫자 위치와 그 숫자는 무엇인가요? 
x = np.where(a > 22000)
type(x)
my_index = x[0][0]
a[my_index]

a[np.where(a > 22000)][0]
a[a > 10000][0]

#처음으로 24000보다 큰 숫자가 나왔을 때, 숫자 위치와 그 숫자는 무엇인가요? 
x = np.where(a > 24000)
type(x)
my_index = x[0][0]
a[my_index]

#처음으로 10000보다 큰 숫자들 중 50번째로 나오는 숫자 위치와 그 숫자는 무엇인가요? 
np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a

x = np.where(a > 10000)
x
my_index = x[0][49]
my_index 
a[my_index]

#500보다 작은 숫자들 중 가장 마지막으로 나오는 숫자 위치와 그 숫자는 무엇인가요? 
np.random.seed(2024)
a = np.random.randint(1, 26346, 1000)
a

x = np.where(a < 500)
x
my_index = x[0][-1]
my_index 
a[my_index]

#빈칸을 나타내는 방법
import numpy as np
a = np.array([20, np.nan, 13, 24, 309])
~np.isnan(a)
a + 3
np.mean(a)
np.nanmean(a)
np.nan_to_num(a, nan = 0)

False
a = None
b = np.nan
b
a
b + 1

a_filtered = a[~np.isnan(a)]
a_filtered

import numpy as np
 str_vec = np.array(["사과", "배", "수박", "참외"])
 str_vec
 str_vec[[0, 2]]

 mix_vec = np.array(["사과", 12, "수박", "참외"], dtype=str)
 mix_vec
 
 combined_vec = np.concatenate((str_vec, mix_vec))
 combined_vec
 
col_stacked = np.column_stack((np.arange(1, 5), np.arange(12, 16)))
col_stacked

row_stacked = np.row_stack((np.arange(1, 5), np.arange(12, 16)))
row_stacked

#길이가 다른 벡터 합치기 
uneven_stacked = np.column_stack((vec1, vec2))
uneven_stacked

vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
vec1


vec1 = np.arange(1, 5)
vec2 = np.arange(12, 18)
vec1 = np.resize(vec1, len(vec2))
vec1

uneven_stacked = np.vstack((vec1, vec2))
uneven_stacked

#연습문제1 주어진 벡터의 각 요소에 5를 더한 새로운 벡터를 생성하세요.
a = np.array([1, 2, 3, 4, 5])
a
a + 5

#연습문제2 주어진 벡터의 홀수 번째 요소만 추출하여 새로운 벡터를 생성하세요.
a = np.array([12, 21, 35, 48, 5])
a%2 == 1
a[0::2]

#연습문제3 주어진 벡터에서 최대값을 찾으세요
 a = np.array([1, 22, 93, 64, 54])
 a.max()
 a

#연습문제4 주어진 벡터에서 중복된 값을 제거한 새로운 벡터를 생성하세요.
a = np.array([1, 2, 3, 2, 4, 5, 4, 6])
np.unique(a)

#연습문제5 주어진 두 벡터의 요소를 번갈아 가면서 합쳐서 새로운 벡터를 생성하세요.
 a = np.array([21, 31, 58])
 b = np.array([24, 44, 67])
 
 #import numpy as np
 #a = np.array([21, np.nan, 31, np.nan, 58, np.nan])
 #a
 #b = np.array([np.nan, 24, np.nan, 44, np.nan, 67])
 #b
 #combined_vec = np.concatenate((a, b))
 #combined_vec

x = np.empty(6)
x

#짝수
x[[1, 3, 5]] = b
x[0::2] = b
x

#홀수
x[[0, 2, 4]] = a
x[0::2] = a
x 



 
 #연습문제6 다음 a 벡터의 마지막 값은 제외한 두 벡터 a와 b를 더한 결과를 구하세요.
 a = np.array([1, 2, 3, 4, 5])
 b = np.array([6, 7, 8, 9])
 
 

