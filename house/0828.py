#####################################################################################
## 이번에는 집값 데이터로 해보기.
#####################################################################################


### 데이터 불러오기
house_train = pd.read_csv("./data/house/train.csv")
house_test = pd.read_csv("./data/house/test.csv")
sub_df = pd.read_csv("./data/house/sample_submission.csv")

### 데이터 알아보기?
house_train.shape
house_test.shape
train_n=len(house_train)

#### train 데이터 전처리 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
#### 데이터 전처리

### NaN 채우기
# 각 숫치형 변수는 평균 채우기
# 각 범주형 변수는 Unknown 채우기

# character
# numeric
## 숫자형 채우기
numeric_train = house_train.select_dtypes(include = [int, float]) # 숫자형 컬럼만 선택해서 데이터프레임으로 가져오기.
numeric_train.isna().sum() # NaN 값이 있는지 확인.
num_selected_train = numeric_train.columns[numeric_train.isna().sum() > 0] # NaN 값이 있는 컬럼 선택해서 컬럼 이름 가져오기.

numeric_train[num_selected_train].isna().sum() # NaN 값이 있는 컬럼만 선택해서 보여줘.

# 평균으로 채우기.
for col in num_selected_train:
    house_train[col].fillna(house_train[col].mean(), inplace=True)

house_train[num_selected_train].isna().sum() # NaN 값 없어진거 확인!

## 범주형 채우기
character_train = house_train.select_dtypes(include = [object]) # 원소가 오브젝트인 컬럼만 가져와서 데이터프레임으로 출력.
character_train.isna().sum() # NaN 값이 있는지 확인.
charac_selected_train = character_train.columns[character_train.isna().sum() > 0] # NaN 값이 있는 컬럼 선택해서 컬럼 이름 가져오기.

character_train[charac_selected_train].isna().sum() # NaN 값이 있는 컬럼만 선택해서 보여줘.

# "최빈값"으로 채우기
for col in charac_selected_train:
    house_train[col].fillna(house_train[col].mode()[0], inplace=True)

house_train[charac_selected_train].isna().sum() # NaN 값 없어진거 확인!
#### train 데이터 전처리 끝 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


#### test 데이터 전처리 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
numeric_test = house_test.select_dtypes(include = [int, float])
numeric_test.isna().sum() # NaN 값이 있는지 확인.
num_selected_test = numeric_test.columns[numeric_test.isna().sum() > 0]

numeric_test[num_selected_test].isna().sum()

# 평균으로 채우기.
for col in num_selected_test:
    house_test[col].fillna(house_test[col].mean(), inplace=True)

house_test[num_selected_test].isna().sum()

## 범주형 채우기
character_test = house_test.select_dtypes(include = [object])
character_test.isna().sum() # NaN 값이 있는지 확인.
charac_selected_test = character_test.columns[character_test.isna().sum() > 0]

character_test[charac_selected_test].isna().sum()

# "최빈값"으로 채우기
for col in charac_selected_test:
    house_test[col].fillna(house_test[col].mode()[0], inplace=True)

house_test[charac_selected_test].isna().sum()
#### test 데이터 전처리 끝 @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

### 통합 df 만들기 + 더미코딩
## 통합 df 만들기
df = pd.concat([house_train, house_test], ignore_index=True)

################################ 나만의 비법 소스
df["MSSubClass"] = df["MSSubClass"].astype(str)
# MSSubClass (집유형)을 문자열로 변경.
################################ 나만의 비법 소스

## 더미코딩
# df.info()
df = pd.get_dummies(
    df,
    columns= df.select_dtypes(include=[object]).columns,
    drop_first=True
    )
df

# train / test 데이터셋
train_df=df.iloc[:train_n,]
test_df=df.iloc[train_n:,]

## x / y 나누기
# train
train_x=train_df.drop(["Id", "SalePrice"], axis=1)
train_y=train_df["SalePrice"]

test_x=test_df.drop(["Id", "SalePrice"], axis=1)

############# 오늘 배운 코드로 체인 벨리데이션 해서 람다값 얻기 ##############
# 교차 검증 설정
kf = KFold(n_splits=5, shuffle=True, random_state=2024)

def rmse(model):
    score = np.sqrt(-cross_val_score(model, train_x, train_y, cv = kf,
                                     scoring = "neg_mean_squared_error").mean())
    return(score)

lasso = Lasso(alpha=0.01) # 코드 잘 만들었는지 테스트
rmse(lasso) # 테스트

# 각 알파 값에 대한 교차 검증 점수 저장
alpha_values = np.arange(0, 1000, 10)
mean_scores = np.zeros(len(alpha_values))

k=0
for alpha in alpha_values:
    lasso = Lasso(alpha=alpha)
    mean_scores[k] = rmse(lasso)
    k += 1

# 결과를 DataFrame으로 저장
df = pd.DataFrame({
    'lambda': alpha_values,
    'validation_error': mean_scores
})

# 결과 시각화
plt.plot(df['lambda'], df['validation_error'], label='Validation Error', color='red')
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.title('Lasso Regression Train vs Validation Error')
plt.show()

# 최적의 alpha 값 찾기
optimal_alpha = df['lambda'][np.argmin(df['validation_error'])]
print("Optimal lambda:", optimal_alpha)
#158.26

### 이제 나온 알파값으로 예상 해보자.
model = Lasso(alpha = 158.26)
model.fit(train_x, train_y)
model.coef_
model.intercept_

test_y_pred = model.predict(test_x)

sub_df["SalePrice"] = test_y_pred #셈플에 가격 넣기
# sub_df.to_csv("../data/houseprice/sample_submission_240828-1.csv", index = False)
# 0.15527