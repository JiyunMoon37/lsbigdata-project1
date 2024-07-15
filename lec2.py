a=1
a
# Ctrl+Enter
# Shift + 화살표 : 블록
# 파워셀 명령어 리스트 
# ls : 파일 목록
# cd : 폴더 이동
# . : 현재 폴더
# .. : 상위 폴더 
# Tab\Shift Tab : 자동완성
# 화면 정리 : cls 

a = 10
a
a = "안녕하세요!"
a = '안녕하세요!'
a = [1, 2, 3]
a
var1 = [1, 2, 3]
var1
var2 = [4, 5, 6]
var2
var1 + var2
str1 = 'a'
str1
str2 = 'text'
str2
str3 = 'Hello World!'
str3
str4 = ['a', 'b', 'c']
str4
a = '안녕하세요!'
a

b = 'LS빅데이터 스쿨!'
b

a+b
a + ' ' +b #문자열 3개가 합쳐짐 

print(a) #내가 출력하고자 하는 값을 더 명확하게 표현함. 

num1 = 3
num2 = 5
num1 + num2 #다른 사람이 내가 만든 변수를 봤을 때 무엇을 의미하는지 알 수 있도록 설정하는 것이 좋음. 

a = 10
b = 3.3
print("a + b =", a + b) #덧셈
print("a - b =", a - b) #뺄셈
print("a * b =", a * b) #곱셈
print("a / b =", a / b) #나눗셈
print("a % b", a % b) #나머지 
print("a // b =", a // b) #몫
print("a ** b =", a ** b) #거듭제곱 
# shift + alt + 아래 화살표 : 아래로 복사 
# Ctrl + alt + 아래 화살표 : 커서 여러개 
(a **6) // 7
(a **6) % 7
(a **6) % 7
(a **6) % 7
(a **6) % 7

a == b
a != b
a < b
a > b
a <= b 
a >= b

# 2에 4승과 12453을 7로 나눈 몫을 더해서 8로 나눴을 때 나머지
a = ((2 ** 4) + (12453 // 7)) % 8
# 9의 7승을 12로 나누고, 36452를 253로 나눈 나머지에 곱한 수
b = ((9 ** 7) / 12) * (36452 % 253)
a > b
a < b

# 사용자 나이 검증 예제
user_age = 25
 is_adult = user_age >= 18
 print("성인입니까?", is_adult)
 
 #False = 3
 #True = 2
 a = "True"
 b = TRUE
 c = true
 d = True 

a = True
b = False

 # and 연산자
print("a and b:", a and b)  # False, a와 b 둘 다 참이어야 참 반환
# or 연산자
print("a or b:", a or b)    
# not 연산자
print("not a:", not a)      
# True, a와 b 중 하나라도 참이면 참 반환
# False, a의 반대 불리언 값 반환

# True : 1로 처리
# False : 0으로 처리 

True + True # 2
True + False # 1
False + False # 0

#and 연산자
True and False
True and True
False and False
False and True

#and는 *으로 환산가능
True * False
True * True
False *   False
False *   True

#or 연산자
True or False
True or True
False or  False
False or  True

a = False
b = False
a or b
min(a + b, 1)

a = 3
a += 10 # a = a + 10
a

a -= 4
a

a %= 3
a

a += 12
a

a **= 2 
a

a /= 7
a

str1  = "hello"
str1 + str1
str1 * 3

# 문자열 변수 할당
str1 = "Hello! "
 # 문자열 반복
repeated_str = str1 * 3
 print("Repeated string:", repeated_str)
 repeated_str
 str1 * 2.5
 
 # 정수 : integer
 # 실수 : float (double) 
 
 # 단항 연산자
x = 5
+x
-x
~x

bin(5)
bin(-5)
bin(-6)
x = -4
~x
bin(-4)
x = 0
~x
x = 0
bin(~x)

!pip install pydataset
import pydataset
pydataset.data()

df = pydataset.data("AirPassengers") #dataframe
df

#테스트
