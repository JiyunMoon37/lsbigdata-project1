import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import uniform
from scipy.stats import t 

#직선의 방정식 
#y=ax + b 
#예 ) y=2x+3의 그래프를 그려보세요 
a = 95
b = 0

x = np.linspace(0, 5, 100)
y = a * x + b

plt.plot(x, y, color = 'blue')
#plt.axvline(0, color = 'black') #수직선
#plt.axhline(0, color = 'black') #수평선 
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
#plt.clf()

house_train = pd.read_csv("./data/house/train.csv")
my_df = house_train[["BedroomAbvGr", "SalePrice"]].head(10)
my_df["SalePrice"] = my_df["SalePrice"] / 1000
plt.scatter(x = my_df["BedroomAbvGr"], y = my_df["SalePrice"])

a = 95
b = 0

x = np.linspace(0, 5, 100)
y = a * x + b

plt.plot(x, y, color = 'blue')
#plt.axvline(0, color = 'black') #수직선
#plt.axhline(0, color = 'black') #수평선 
# plt.xlim(-10, 10)
# plt.ylim(-10, 10)

plt.show()
plt.clf()


#점들에 가까워지게 그래프 설정
#곡선인 경우
a = 5
b = 100
x = np.linspace(0,5, 100)
y = np.exp(x) * a + b

#직선인 경우
a1 = 80
b1 = -30
y = a1 * x + b1

plt.clf()
my_df = train_df[['BedroomAbvGr','SalePrice']].head(10)
my_df['SalePrice'] = my_df['SalePrice']/1000
plt.scatter(x=my_df['BedroomAbvGr'], y=my_df['SalePrice'], color='orange')
plt.plot(x , y)
plt.ylim([-1,400])
plt.show()

#쌤 
#테스트 집 정보 가져오기
house_test = pd.read_csv("./data/house/test.csv")
a = 80
b = 30

#sub 데이터 불러오기 
sub_df = pd.read_csv("./data/house/sample_submission.csv")
sub_df

#SalePrice 바꿔치기 
sub_df["SalePrice"] =(a * house_test["BedroomAbvGr"] + b) * 1000
sub_df

sub_df.to_csv("./data/house/sample_submission3.csv", index = False) 

#우리조
test = pd.read_csv("data/houseprice/test.csv")
test = test.assign(SalePrice = ((80* test["BedroomAbvGr"] -30) * 1000))
test["SalePrice"]

sample_submission = pd.read_csv("data/houseprice/sample_submission.csv")
sample_submission["SalePrice"] = test["SalePrice"]


#직선 성능 평가
a = 70
b =10

#y_hat은 어떻게 구할까?
house_train = pd.read_csv("./data/house/train.csv")
y_hat = (a * house_train["BedroomAbvGr"] +b) * 1000

#y는 어디에 있는가?
y = house_train["SalePrice"]


np.abs(y - y_hat) #절대거리 
np.sum(np.abs(y - y_hat))




!pip install scikit-learn
#필요한 패키지 불러오기 
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#필요한 데이터 불러오기 
house_train = pd.read_csv("./data/house/train.csv")
house_test = pd.read_csv("./data/house/test.csv")
sub_df = pd.read_csv("./data/house/sample_submission.csv")

my_df = house_train[['BedroomAbvGr','SalePrice']]

#회귀분석 적합(fit)하기 
x = np.array(house_train["BedroomAbvGr"]).reshape(-1,1)
y = house_train["SalePrice"] 

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) #지표를 가장 최소로 만드는 x와 y값 , 자동으로 기울기, 절편 값을 구해줌. 

# 회귀 직선의 기울기와 절편
model.coef_    #기울기 a (방 하나가 증가할때 증가하는 가격 - 16.38이므로 1600만원 증가한다고 생각 
model.intercept_ #절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산
y_pred = model.predict(x) #y_hat 값 (직선식에 x값 넣는다.) (x는 방갯수)

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x, y_pred, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()





##0802 1교시 

# 회귀직선 구하기
import numpy as np
from scipy.optimize import minimize


def my_f(x):
  return x**2 + 3

my_f(3)


# 초기 추정값
initial_guess = [0]

# 최소값 찾기
result = minimize(my_f, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)



#z = x**2 + y**2 + 3
def my_f2(x):
    return x[0]**2 + x[1]**2 + 3

my_f2([1, 3]) #리스트 형식으로 

# 초기 추정값
initial_guess = [-10, 3]

# 최소값 찾기
result = minimize(my_f2, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)


#f(x,y,z) = (x-1)^2 + (y-2)^2 + (z-4)^2 +7
def my_f3(x):
    return ((x[0]-1)**2 + (x[1]-2)**2 + (x[2]-4)**2 + 7)

my_f3([1, 2, 3]) #리스트 형식으로 

# 초기 추정값
initial_guess = [-10, 3, 4]

# 최소값 찾기
result = minimize(my_f3, initial_guess)

# 결과 출력
print("최소값:", result.fun)
print("최소값을 갖는 x 값:", result.x)














#0802 2교시 
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#필요한 데이터 불러오기
house_train = pd.read_csv("./data/house/train.csv")
house_test = pd.read_csv("./data/house/test.csv")
sub_df = pd.read_csv("./data/house/sample_submission.csv")

#회귀분석 적합(fit)하기
x = np.array(house_train["BedroomAbvGr"]).reshape(-1,1)
y = house_train["SalePrice"]/1000

# 선형 회귀 모델 생성

model = LinearRegression()

# 모델 학습

model.fit(x, y) #지표를 가장 최소로 만드는 x와 y값 , 자동으로 기울기, 절편 값을 구해줌.

# 회귀 직선의 기울기와 절편

model.coef_    #기울기 a (방 하나가 증가할때 증가하는 가격 - 16.38이므로 1600만원 증가한다고 생각
model.intercept_ #절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산

y_pred = model.predict(x) #y_hat 값 (직선식에 x값 넣는다.) (x는 방갯수)

test_x = np.array(house_test["BedroomAbvGr"]).reshape(-1,1)
test_x

pred_y = model.predict(test_x) #test 셋에 대한 집값
pred_y

#sub 데이터 불러오기
sub_df = pd.read_csv("./data/house/sample_submission.csv")
sub_df

#SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y*1000
sub_df

#내보내기
sub_df.to_csv("./data/house/sample_submission4.csv", index = False)









#### 주영이 간단 방법
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#필요한 데이터 불러오기
house_train = pd.read_csv("./data/house/train.csv")
house_test = pd.read_csv("./data/house/test.csv")
sub_df = pd.read_csv("./data/house/sample_submission.csv")


train_x = np.array(house_train["GrLivArea"]).reshape(-1, 1) #시리즈에선 reshape이 안됨
train_y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(train_x, train_y) # 자동으로 기울기, 절편 값을 구해줌

# 예측값 계산
y_pred_train = model.predict(train_x)
y_pred_train


#이상치 탐색
house_train.query("GrLivArea<=4500")

#회귀분석 적합하기 
test_x = np.array(house_test["GrLivArea"]).reshape(-1, 1) #시리즈에선 reshape이 안됨

# 예측값 계산 - 쌤 pred_y
y_pred_test = model.predict(test_x)
y_pred_test

# #SalePrice 바꿔치기
# sub_df["SalePrice"] = y_pred_test
# sub_df

# 회귀 직선의 기울기와 절편
model.coef_    #기울기 a (방 하나가 증가할때 증가하는 가격 - 16.38이므로 1600만원 증가한다고 생각 
model.intercept_ #절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# # 예측값 계산
# y_pred = model.predict(x) #y_hat 값 (직선식에 x값 넣는다.) (x는 방갯수)

# 데이터와 회귀 직선 시각화
plt.scatter(train_x, train_y, color='blue', label='data')
plt.plot(test_x , y_pred_test, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
plt.clf()



#내보내기 
sub_df.to_csv("./data/house/sample_submission5.csv", index = False) 

--
# 쌤
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#필요한 데이터 불러오기
house_train = pd.read_csv("./data/house/train.csv")
house_test = pd.read_csv("./data/house/test.csv")
sub_df = pd.read_csv("./data/house/sample_submission.csv")

#이상치 탐색
house_train = house_train.query("GrLivArea<=4500")

 # house_train["GrLivArea"] #pandas series 
 # house_train[["GrLivArea"]] #pandas frame 

#회귀분석 적합하기 
x = house_train[["GrLivArea", "GarageArea"]] #시리즈에선 reshape이 안됨 #열이 2개 #대괄호 2개 #np.array쓰는 이유는 열 한개짜리 쓰기위해서 
y = house_train["SalePrice"]

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_    #기울기 a (방 하나가 증가할때 증가하는 가격 - 16.38이므로 1600만원 증가한다고 생각 
model.intercept_ #절편 b

slope = model.coef_[0]
intercept = model.intercept_
print(f"기울기 (slope): {slope}")
print(f"절편 (intercept): {intercept}")

# 예측값 계산 - 쌤 pred_y
pred_y = model.predict(x)
pred_y

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x , pred_y, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 5000])
plt.ylim([0, 900000])
plt.legend()
plt.show()
plt.clf()

x = np.array(house_test["GrLivArea"]).reshape(-1, 1) #시리즈에선 reshape이 안됨

# 예측값 계산 - 쌤 pred_y
pred_y = model.predict(x)
pred_y

sub_df["SalePrice"] = pred_y

#내보내기 
sub_df.to_csv("./data/house/sample_submission6.csv", index = False) 







##4교시
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#필요한 데이터 불러오기
house_train = pd.read_csv("./data/house/train.csv")
house_test = pd.read_csv("./data/house/test.csv")
sub_df = pd.read_csv("./data/house/sample_submission.csv")

#이상치 탐색
house_train = house_train.query("GrLivArea<=4500")

#회귀분석 적합하기 
#숫자형 변수만 선택하기
x = house_train.select_dtypes(include=[int, float])

#필요없는 칼럼 제거하기 
x = x.iloc[:,1:-1] 
y = house_train["SalePrice"]
x.isna().sum()

#평균값을 대체하기
x.fillna(x.mean(), inplace = True) #내가 추가함 
x.isna().sum()

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_    #기울기 a (방 하나가 증가할때 증가하는 가격 - 16.38이므로 1600만원 증가한다고 생각 
model.intercept_ #절편 b
model.coef_[0]
model.coef_[1]

# slope = model.coef_[0]
# intercept = model.intercept_
# print(f"기울기 (slope): {slope}")
# print(f"절편 (intercept): {intercept}")

def my_houseprice(x,y) : 
    return model.coef_[0]*x + model.coef_[1]*y + model.intercept_

temp_result = my_houseprice(house_test["GrLivArea"], house_test["GarageArea"])
# temp_result.isna.sum()

my_houseprice(house_test["GrLivArea"], house_test["GarageArea"])

test_x = house_test[["GrLivArea", "GarageArea"]]
test_x

#결측치 확인
test_x["GrLivArea"].isna().sum() #결측치 없음
test_x["GarageArea"].isna().sum() #결측치 하나 있음 
test_x = test_x.fillna(house_test["GarageArea"].mean())

#test_x.fillna(house_test["GarageArea"].mean(), inplace=True) #x에 다시 반영안해도 되는 옵션 

# 예시: train_x에서 사용한 피처를 test_x에도 동일하게 적용
train_columns = x.columns  # 학습할 때 사용한 열
test_x = test_x[train_columns]  # test_x를 train_columns에 맞게 필터링



# 예측값 계산 - 쌤 pred_y
pred_y = model.predict(test_x)
pred_y

# 데이터와 회귀 직선 시각화
plt.scatter(x, y, color='blue', label='data')
plt.plot(x , pred_y, color='red', label='regression')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0, 5000])
plt.ylim([0, 900000])
plt.legend()
plt.show()
plt.clf()

x = np.array(house_test["GrLivArea"]).reshape(-1, 1) #시리즈에선 reshape이 안됨

# 예측값 계산 - 쌤 pred_y
pred_y = model.predict(test_x)
pred_y

#SalePrice 바꿔치기 
sub_df["SalePrice"] = pred_y

#내보내기 
sub_df.to_csv("./data/house/sample_submission7.csv", index = False) 


###0805 2교시
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#필요한 데이터 불러오기
house_train = pd.read_csv("./data/house/train.csv")
house_test = pd.read_csv("./data/house/test.csv")
sub_df = pd.read_csv("./data/house/sample_submission.csv")

#이상치 탐색
house_train = house_train.query("GrLivArea<=4500")

#회귀분석 적합하기 
#숫자형 변수만 선택하기
x = house_train.select_dtypes(include=[int, float])

#필요없는 칼럼 제거하기 
x = x.iloc[:,1:-1] 
y = house_train["SalePrice"]
x.isna().sum()

# #변수별로 결측값 채우기
# fill_values = {
#     'LotFrontage' : x["LotFrontage"].mean(),
#     'MasVnrArea' : x["MasVnrArea"].mean(),
#     'GarageYrBlt' : x["GarageYrBlt"].mean()
# }

#쌤 
x = x.fillna(value = fill_values)
x.isna().sum()

# #평균값을 대체하기
# x.fillna(x.mean(), inplace = True) #내가 추가함 
# x.isna().sum()

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y) # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_    #기울기 a (방 하나가 증가할때 증가하는 가격 - 16.38이므로 1600만원 증가한다고 생각 
model.intercept_ #절편 b
model.coef_[0]
model.coef_[1]

# slope = model.coef_[0]
# intercept = model.intercept_
# print(f"기울기 (slope): {slope}")
# print(f"절편 (intercept): {intercept}")

test_x =house_test.select_dtypes(include=[int, float])

#필요없는 칼럼 제거하기 
test_x = test_x.iloc[:,1:] 
test_x.isna().sum()

# #변수별로 결측값 채우기
# fill_values = {
#     'LotFrontage' : test_x["LotFrontage"].mean(),
#     'MasVnrArea' : test_x["MasVnrArea"].mean(),
#     'BsmtFinSF1' : test_x["BsmtFinSF1"].mean(),
#     'BsmtFinSF2' : test_x["BsmtFinSF2"].mean(),
#     'BsmtUnfSF' : test_x["BsmtUnfSF"].mean(),
#     'TotalBsmtSF' : test_x["TotalBsmtSF"].mean(),
#     'BsmtFullBath' : test_x["BsmtFullBath"].mean(),
#     'BsmtHalfBath' : test_x["BsmtHalfBath"].mean(),
#     'GarageYrBlt' : test_x["GarageYrBlt"].mean(),
#     'GarageCars' : test_x["GarageCars"].mean(),
#     'GarageArea' : test_x["GarageArea"].mean()
# }

#쌤 
test_x = test_x.fillna(value = fill_values)
test_x.isna().sum()

# test_x.fillna(test_x.mean(), inplace = True) #내가 추가함 
# test_x.isna().sum()

# 예측값 계산 - 쌤 pred_y
pred_y = model.predict(test_x)
pred_y

#SalePrice 바꿔치기 
sub_df["SalePrice"] = pred_y

#내보내기 
sub_df.to_csv("./data/house/sample_submission8.csv", index = False) 

###0805 4교시
# 필요한 패키지 불러오기
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

## 필요한 데이터 불러오기
house_train=pd.read_csv("./data/houseprice/train.csv")
house_test=pd.read_csv("./data/houseprice/test.csv")
sub_df=pd.read_csv("./data/houseprice/sample_submission.csv")

## 이상치 탐색
# house_train=house_train.query("GrLivArea <= 4500")


## 회귀분석 적합(fit)하기
# house_train["GrLivArea"]   # 판다스 시리즈
# house_train[["GrLivArea"]] # 판다스 프레임
# 숫자형 변수만 선택하기
x = house_train.select_dtypes(include=[int, float])
# 필요없는 칼럼 제거하기
x = x.iloc[:,1:-1]
y = house_train["SalePrice"]


# 변수별로 결측값 채우기
fill_values = {
    'LotFrontage': x["LotFrontage"].mean(),
    'MasVnrArea': x["MasVnrArea"].mean(),
    'GarageYrBlt': x["GarageYrBlt"].mean()
}
x = x.fillna(value=fill_values)
x.isna().sum()
x.mean()

# 선형 회귀 모델 생성
model = LinearRegression()

# 모델 학습
model.fit(x, y)  # 자동으로 기울기, 절편 값을 구해줌

# 회귀 직선의 기울기와 절편
model.coef_      # 기울기 a
model.intercept_ # 절편 b

# 테스트 데이터 예측
test_x = house_test.select_dtypes(include=[int, float])
test_x = test_x.iloc[:,1:]

# fill_values = {
#     'LotFrontage': test_x["LotFrontage"].mean(),
#     'MasVnrArea': test_x["MasVnrArea"].mode()[0],
#     'GarageYrBlt': test_x["GarageYrBlt"].mode()[0]
# }
# test_x = test_x.fillna(value=fill_values)
test_x=test_x.fillna(test_x.mean())


# 결측치 확인
test_x.isna().sum()
test_x.mean()
# 테스트 데이터 집값 예측
pred_y=model.predict(test_x) # test 셋에 대한 집값
pred_y

# SalePrice 바꿔치기
sub_df["SalePrice"] = pred_y
sub_df

# csv 파일로 내보내기
sub_df.to_csv("./data/house/sample_submission9.csv", index=False)

##0.22086나옴. 

















