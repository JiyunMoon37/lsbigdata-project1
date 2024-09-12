def g(x=3):
    result = x + 1
    return result
g()  

print(g)

#함수 내용확인
import inspect
print(inspect.getsource(g))

#if ... else 정식 
x = 3
if x > 4:
    y = 1
else:
    y = 2
print(y)

#if else 축약 
y = 1 if x > 4 else 2

#리스트 컴프리헨션
x = [1, -2, 3, -4, 5]
result = ["양수" if value > 0 else "음수" for value in x]
print(result)

#조건 3개 이상의 경우 elif()
x = 0
 if x > 0:
    result = "양수"
 elif x == 0:
    result = "0"
 else:
    result = "음수"
 print(result)


#조건 3가지 넘파이 버전
import numpy as np 
x = np.array([1, -2, 3, -4, 0])
conditions = [x > 0, x == 0, x < 0]
choices = ["양수", "0", "음수"]
result = np.select(conditions, choices, x)
print(result)

#for loop 
for i in range(1, 4):
    print(f"Here is {i}")

#for loop 리스트 컴프
print([f"Here is {i}" for i in range(1, 4)])

name = "John"
age = 30
greeting = f"Hello, my name is {name} and I am {age} years old."
print(greeting)

import numpy as np
names = ["John", "Alice"]
ages = np.array([25, 30]) 

greetings = [f"Hello, my name is {name} and I am {age} years old." for name, age
 in zip(names, ages)]

#0830 4교시 
#while 문
i = 0
while i <= 10:
    i += 3
    print(i)

#while, break 문
i = 0
while True:
    i += 3
    if i > 10:
        break
    print(i)

#apply 
 import pandas as pd

 data = pd.DataFrame({
    'A': [1, 2, 3], 'B': [4, 5, 6]
 })
 data 

df.apply(max, axis=0)
df.apply(max, axis=1)

def my_func(x, const=3):
    return max(x)**2 + const

my_func([3, 4, 10], 5)

data.apply(my_func, axis = 0, const=5)

import numpy as np
array_2d = np.arange(1, 13).reshape((3, 4), order='F')
print(array_2d)

np.apply_along_axis(max, axis=0, arr=array_2d)

#함수 환경
y = 2

def my_func(x) :
    global y 

    def my_f(k) :
       return k**2
    
    y = my_f(x) + 1 
    result = x + y

    return result

my_func(3)
print(y) 

#입력값이 몇 개일지 모를땐 
def add_many(*args): 
     result = 0 
     for i in args: 
         result = result + i   # *args에 입력받은 모든 값을 더한다.
     return result 

add_many(1, 2, 3)

def first_many(*args):
   return args[0]

first_many(1, 2, 3)
first_many(4, 1, 2, 3)

#0830 5교시 
def add_mul(choice, *args): 
     if choice == "add":   # 매개변수 choice에 "add"를 입력받았을 때
         result = 0 
         for i in args: 
             result = result + i 
     elif choice == "mul":   # 매개변수 choice에 "mul"을 입력받았을 때
         result = 1 
         for i in args: 
             result = result * i 
     return result 

add_mul("mul", 5, 4, 3, 1)

#별표 2개 (**)는 입력값을 딕셔너리로 만들어줌!
def my_twostars(choice, **kwargs):
    if choice == "first":
        return print(kwargs[0])
    elif choice == "second":
        return print(kwargs[1])
    else :
        return print(kwargs)
    
my_twostars("first", age = 30, name = "issac")

dict_a = {'age' : 30, 'name' : 'issac'}
dict_a["age"]
dict_a["name"]
