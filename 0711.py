# 데이터 타입
 x = 15.34
 print(x, "는 ", type(x), "형식입니다.", sep='')
 
  # 문자형 데이터 예제
a = "Hello, world!"
b = 'python programming'

 print(a, type(a))
 print(b, type(b))
 
  # 여러 줄 문자열
ml_str = """This is
 a multi-line
 string"""

 print(ml_str)
 print(ml_str, type(ml_str))
 
 #문자열 결합
 greeting = "안녕" + " " + "파이썬!"
 print("결합된 문자열:", greeting)
 
 #리스트
 fruit = ["apple", "banana", "cherry"]
 type(fruit)
 
 numbers = [1, 2, 3, 4, 5]
 type(numbers)
 
 mixed_list = [1, "Hello", [1, 2, 3]]
 type(mixed_list)
 
 #튜플형데이터타입
 a_tp = (10, 20, 30, 40, 50) 
 a_tp[3:]
 a_tp[:3]
 a_tp[1:3]
 
 a_tp[2]
 a_tp[1]= - 25
 a_tp
 
 b_int = (10)
 b_int
 b_tp = (42,)
 b_tp
 
 type(b_int)
 type(b_tp)
 
 
a_ls = [10, 20, 30, 40, 50]
a_ls[1:4]
a_ls[1] = 25
a_ls

#사용자 정의 함수 
 def min_max(numbers):
 return min(numbers), max(numbers)

a = [1, 2, 3, 4, 5]
result = min_max(a)
type(result) 
 result = min_max((1, 2, 3, 4, 5))
 print("Minimum and maximum:", result) 
 
 #딕셔너리 생성 예제
person = {
   'name' : 'John',
   'age' : 30,
   'city': 'New York'
 }
print("Person:", person) 

jiyun = {
   "이름" : "Jiyun",
   "나이" : (26,20),
   "사는 곳": ["서울", "부산"]
 }
print("Jiyun:", jiyun) 

jiyun_age = jiyun.get('나이')
jiyun_age[0]

#집합
fruits = {'apple', 'banana', 'cherry', 'apple'}
print("Fruits set:", fruits)
type(fruits)

#빈 집합 생성
empty_set = set()
print("Empty set:", empty_set)

empty_set.add("apple")
empty_set.add("banana")
empty_set.add("apple")
empty_set

empty_set.discard("orange")
empty_set

#집합 간 연산
other_fruits = {'berry', 'cherry'}
union_fruits = fruits.union(other_fruits) #합집합 
intersection_fruits = fruits.intersection(other_fruits) #교집합 
print("Union of fruits:", union_fruits)
print("Intersection of fruits:", intersection_fruits)

 #논리형 데이터 예제
p = True
q = False
print(p, type(p))
print(q, type(q))
print(p + p)  # True는 1로, False는 0으로 계산됩니다

is_active = True
is_greater = 10 > 5  # True 반환
is_equal = (10 == 5)  # False 반환
print("Is active:", is_active)
print("Is 10 greater than 5?:", is_greater)
print("Is 10 equal to 5?:", is_equal)

a=3
 if (a == 2):
 print("a는 2와 같습니다.")
 else:
 print("a는 2와 같지 않습니다.")
 
 #숫자형을 문자열형으로 변환
num = 123
str_num = str(num)
print("문자열:", str_num, type(str_num))

#문자열형을 숫자형(실수)으로 변환
num_again = float(str_num)
print("숫자형:", num_again, type(num_again))

#리스트와 튜플 변환
lst = [1, 2, 3]
print("리스트:", lst)
tup = tuple(lst)
print("튜플:", tup)

set_example = {'a', 'b', 'c'} #{}로 설정해줘서 집합임. 
 #딕셔너리로 변환 시, 일반적으로 집합 요소를 키 또는 값으로 사용
dict_from_set = {key: True for key in set_example}
 print("Dictionary from set:", dict_from_set)
 
#흰 색 책 p63_seaborn 패키지 설치
#!pip install seaborn
import seaborn as sns #약어
import matplotlib.pyplot as plt 
 
#var = ["a", "a", "b", "c"]
#var
 
#sns.countplot(x=var)
#plt.show()
#plt.clf() #그 전에 설정값 초기화 

plt.clf() #그 전에 설정값 초기화 
df = sns.load_dataset("titanic")
sns.countplot(data = df, x = "sex")
sns.countplot(data = df, x = "sex", hue= "sex")
plt.show()

#?sns.countplot
plt.clf() #그 전에 설정값 초기화 
#sns.countplot(data=df, x = "class")
#sns.countplot(data=df, x = "class", hue= "alive")
sns.countplot(data=df, x = "class", hue= "alive", orient="v")
plt.show()



import sklearn.metrics
sklearn.metrics.accuracy_score()

from sklearn import metrics
#metrics.accurcy_score()

import sklearn.metrics as met 
met.accuracy_score()

from sklearn import metrics as met
met.accuracy_score()

from sklearn.metrics import accuracy_score as accuracy
accuracy()
