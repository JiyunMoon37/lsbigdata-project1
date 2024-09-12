# 필요한 라이브러리 설치
!pip install pandas statsmodels scikit-learn

import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np
from sklearn.model_selection import train_test_split

## 문제 1 : 데이터를 로드하고, 로지스틱 회귀모델을 적합하고, 회귀 표를 작성하세요.
# 파일을 판다스로 읽기
file_path ="C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project1/data/leukemia_remission.txt"

# 공백 또는 탭으로 구분된 데이터를 pandas DataFrame으로 불러오기
data = pd.read_csv(file_path, sep='\t') # 공백으로 구분된 파일
#data =  pd.read_table("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project1/data/leukemia_remission.txt", delimiter='\t')
print(data.head())

# 종속 변수와 독립 변수 설정
X = data[['CELL', 'SMEAR', 'INFIL', 'LI', 'BLAST', 'TEMP']]
y = data['REMISS']

# 상수항 추가
X = sm.add_constant(X)

# 로지스틱 회귀 모델 적합
model = sm.Logit(y, X).fit()

# 회귀 표 출력
print(model.summary())

## 문제 2 : 해당 모델은 통계적으로 유의한가요? 그 이유를 검정통계량를 사용해서 설명하시오.
# p-값 확인
print(model.pvalues)

#const    0.391348
#CELL     0.554288
#SMEAR    0.688249
#INFIL    0.702039
#LI       0.100899
#BLAST    0.995940
#TEMP     0.197623

## 문제 3 : 유의수준이 0.2를 기준으로 통계적으로 유의한 변수는 몇개이며, 어느 변수 인가요?
# 유의한 변수 확인
# 유의수준 0.2 기준으로 통계적으로 유의한 변수 찾기
significant_vars = model.pvalues[model.pvalues <= 0.2]
print("유의수준 0.2 이하의 통계적으로 유의한 변수:")
print(significant_vars)

#LI      0.100899
#TEMP    0.197623

## 문제 4: 환자 오즈 계산
# 환자 데이터 입력 (주어진 값)
patient_data = [0.65, 0.45, 0.55, 1.2, 1.1, 0.9]  # CELL, SMEAR, INFIL, LI, BLAST, TEMP

# 절편 및 회귀 계수 가져오기
intercept = model.params[0]  # 절편
coefficients = model.params[1:]  # 회귀 계수

# 오즈 계산
odds = np.exp(intercept + np.dot(coefficients, patient_data))

print("환자의 오즈:", odds)

#오즈 : 환자의 오즈: 0.03817043186641846

## 문제 5: 위 환자의 혈액에서 백혈병 세포가 관측되지 않은 확률은 얼마인가요?
# 오즈 계산
odds = np.exp(model.predict(X))
non_leukemia_probability = 1 - odds

print("환자의 혈액에서 백혈병 세포가 관측되지 않은 확률:", non_leukemia_probability.mean())

#환자의 혈액에서 백혈병 세포가 관측되지 않은 확률: -0.4608753878207472

# 문제 6: TEMP 변수의 계수는 얼마이며, 해당 계수를 사용해서 TEMP 변수가 백혈병 치료에 대한 영향을 설명하시오.
temp_coefficient = model.params['TEMP']
print("TEMP 변수의 계수:", temp_coefficient)

# TEMP 변수의 영향 설명
if temp_coefficient > 0:
    print("TEMP 변수는 백혈병 치료에 긍정적인 영향을 미친다.")
else:
    print("TEMP 변수는 백혈병 치료에 부정적인 영향을 미친다.")

#TEMP 변수의 계수: -100.17340003287605
#TEMP 변수는 백혈병 치료에 부정적인 영향을 미친다.


## 문제 7: CELL 변수의 99% 오즈비에 대한 신뢰구간

# CELL 변수의 계수 및 표준 오차 가져오기
cell_coefficient = model.params['CELL']
cell_standard_error = model.bse['CELL']

# 오즈비 계산
cell_odds_ratio = np.exp(cell_coefficient)

# 99% 신뢰구간 계산
z_score = sm.stats.zconfint(0.99)  # 99% 신뢰수준에 대한 z-값
lower_bound = np.exp(cell_coefficient - z_score * cell_standard_error)
upper_bound = np.exp(cell_coefficient + z_score * cell_standard_error)

print("CELL 변수의 오즈비:", cell_odds_ratio)
print("99% 신뢰구간:", (lower_bound, upper_bound))


#CELL 변수의 99% 오즈비 신뢰구간: 
#0    1.027206e-31
#1    5.847804e+57

## 문제 8 : 주어진 데이터에 대하여 로지스틱 회귀 모델의 예측 확률을 구한 후, 50% 이상인 경우 1로 처리하여, 혼동 행렬를 구하시오.
# 예측 확률
pred_probs = model.predict(X)
predictions = (pred_probs >= 0.5).astype(int)

# 혼동 행렬
conf_matrix = confusion_matrix(y, predictions)
print("혼동 행렬:\n", conf_matrix)

#혼동 행렬:
 #[[15  3]
 #[ 4  5]]

## 문제 9 : 해당 모델의 Accuracy는 얼마인가요?
# Accuracy 계산
accuracy = accuracy_score(y, predictions)
print(f"모델의 Accuracy: {accuracy}")

#모델의 Accuracy: 0.7407407407407407

## 문제 10 : 해당 모델의 F1 Score를 구하세요.
# F1 Score 계산
f1 = f1_score(y, predictions)
print(f"F1 Score: {f1}")

#F1 Score: 0.5882352941176471 
