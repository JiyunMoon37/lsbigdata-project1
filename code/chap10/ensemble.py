from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bagging_model = BaggingClassifier(DecisionTreeClassifier(), 
                                  n_estimate = 100,
                                  max_sample = 100,
                                  n_jobs = 1, random_stare = 42)
                                  
# * n_estimator : Bagging에 사용될 모델 개수
# * max_sample : 데이터셋 만들 때 뽑을 표본크기 

# bagging_model.fit(X_train, y_train)

##0910 4시
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators = 50, 
                                max_leaf_node = 16, 
                                n_jobs = -1, )
