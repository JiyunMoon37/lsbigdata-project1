## 문제 1: 데이터 로드 및 로지스틱 회귀 모델 적합
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import f1_score

# 데이터 로드
# 파일을 판다스로 읽기
file_path ="C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project1/data/leukemia_remission.txt"

# 공백 또는 탭으로 구분된 데이터를 pandas DataFrame으로 불러오기
data = pd.read_csv(file_path, sep='\t') # 공백으로 구분된 파일
#data =  pd.read_table("C:/Users/USER/Documents/LS빅데이터스쿨/lsbigdata-project1/data/leukemia_remission.txt", delimiter='\t')
print(data.head())

# 독립 변수와 종속 변수 정의
X = data[['CELL', 'SMEAR', 'INFIL', 'LI', 'BLAST', 'TEMP']]
y = data['REMISS']

# 상수항 추가
X = sm.add_constant(X)

# 로지스틱 회귀 모델 적합
model = sm.Logit(y, X).fit()

# 회귀 표 작성
print(model.summary())

## 문제 2: 모델의 통계적 유의성 검정
# 모델의 유의성 평가
# p-value 확인
p_values = model.pvalues
print("\n각 변수의 p-value:")
print(p_values)

# 유의성 판단
alpha = 0.05
significant_vars = p_values[p_values < alpha].index.tolist()
if significant_vars:
    print(f"\n유의한 변수들: {significant_vars}")
else:
    print("\n유의한 변수가 없습니다.")

# 예측 수행
# 예측 확률 계산
predicted_probabilities = model.predict(X)

# 예측 클래스 결정 (임계값 0.5 사용)
predicted_classes = (predicted_probabilities >= 0.5).astype(int)

# 예측 결과 출력
data['Predicted_Probabilities'] = predicted_probabilities
data['Predicted_Classes'] = predicted_classes

print("\n예측 결과:")
print(data[['REMISS', 'Predicted_Probabilities', 'Predicted_Classes']].head())

# 혼동 행렬 출력
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(data['REMISS'], predicted_classes)
print("\n혼동 행렬:")
print(conf_matrix)

# 정확도 계산
accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
print(f"\n모델의 정확도: {accuracy:.2f}")

## 문제 3: 유의수준 0.2 기준으로 통계적으로 유의한 변수
# 유의수준 0.2 기준으로 유의한 변수 확인
alpha = 0.2
significant_vars_0_2 = p_values[p_values < alpha].index.tolist()

print(f"\n유의수준 0.2에서 유의한 변수들: {significant_vars_0_2}")
print(f"유의한 변수의 개수: {len(significant_vars_0_2)}")

#유의수준 0.2에서 유의한 변수들: ['LI', 'TEMP']
#유의한 변수의 개수: 2

## 문제 4: 주어진 환자에 대한 오즈 계산
# 환자 특성 정의
patient_data = {
    'const': 1,  # 상수항
    'CELL': 0.65,
    'SMEAR': 0.45,
    'INFIL': 0.55,
    'LI': 1.2,
    'BLAST': 1.1,
    'TEMP': 0.9
}

# 오즈 계산
log_odds = model.params.dot(pd.Series(patient_data))
odds = np.exp(log_odds)

print(f"\n환자의 오즈: {odds:.4f}")

# 환자의 오즈: 0.0382

## 문제 5: 백혈병 세포가 관측되지 않은 확률 계산
# 예측 확률 계산
predicted_probability = model.predict(pd.Series(patient_data))
non_observed_probability = 1 - predicted_probability

print(f"\n백혈병 세포가 관측되지 않은 확률: {non_observed_probability.values[0]:.4f}")

#백혈병 세포가 관측되지 않은 확률: 0.9632

## 문제 6: TEMP 변수의 계수와 설명
# TEMP 변수의 계수
temp_coefficient = model.params['TEMP']
print(f"\nTEMP 변수의 계수: {temp_coefficient:.4f}")

# TEMP의 영향 설명
print("TEMP 변수의 계수는 치료에 대한 영향력을 나타냅니다. 계수가 양수인 경우, TEMP가 증가할수록 백혈병 치료의 성공 확률이 증가하는 것을 의미합니다.")

#TEMP 변수의 계수: -100.1734
#TEMP 변수의 계수는 치료에 대한 영향력을 나타냅니다. 
#계수가 양수인 경우, TEMP가 증가할수록 백혈병 치료의 성공 확률이 증가하는 것을 의미합니다.

## 문제 7: CELL 변수의 99% 오즈비에 대한 신뢰구간
import statsmodels.api as sm

# CELL 변수의 오즈비 및 신뢰구간 계산
cell_odds_ratio = np.exp(model.params['CELL'])

# 99% 신뢰구간 계산
conf_int = model.conf_int(alpha=0.01).loc['CELL']  # alpha=0.01로 99% 신뢰구간
cell_conf_int = np.exp(conf_int)

print(f"CELL 변수의 오즈비: {cell_odds_ratio:.4f}")
print(f"CELL 변수의 99% 신뢰구간: ({cell_conf_int[0]:.4e}, {cell_conf_int[1]:.4e})")

#CELL 변수의 99% 신뢰구간: (1.1673e-45, 5.1460e+71)

## 문제 8: 로지스틱 회귀 모델의 예측 확률과 혼동 행렬
# 예측 확률 계산
data['Predicted_Probabilities'] = model.predict(X)
data['Predicted_Classes'] = (data['Predicted_Probabilities'] >= 0.5).astype(int)

# 혼동 행렬 계산
conf_matrix = confusion_matrix(data['REMISS'], data['Predicted_Classes'])
print("\n혼동 행렬:")
print(conf_matrix)

#혼동 행렬:
#[[15  3]
#[ 4  5]]

## 문제 9: 모델의 Accuracy 계산
# 정확도 계산
accuracy = (conf_matrix[0, 0] + conf_matrix[1, 1]) / conf_matrix.sum()
print(f"\n모델의 Accuracy: {accuracy:.2f}")

#모델의 Accuracy: 0.74

## 문제 10: F1 Score 계산 
# F1 Score 계산
f1 = f1_score(data['REMISS'], data['Predicted_Classes'])
print(f"\n모델의 F1 Score: {f1:.4f}")

#모델의 F1 Score: 0.5882

