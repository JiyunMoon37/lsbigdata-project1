import numpy as np

mat_a = np.array([14, 4, 0, 10]).reshape(2,2)
mat_a

#귀무가설 : 두 변수 독립
#대립가설 : 두 변수가 독립 x 
from scipy.stats import chi2_contingency

chi2, p, df, expected = chi2_contingency(mat_a, correction = True)
chi2.round(3) #검정 통계량
p.round(4) #p-value 

np.sum((mat_a - expected)**2 / expected) 

#유의수준 0.05이라면,
#p값이 0.05보다 작으므로, 귀무가설을 기각 (두 변수 간의 영향이 있다.)
#즉, 두 변수는 독립 아니다. 
#X ~ chi2(1) 일 때, P(X > 15.556)=?
from scipy.stats import chi2

1-chi2.cdf(15.556, df=1)
p

#0903 3교시 
#귀무가설 : 두 도시에서의 음료 선호도가 동일하다.
#대립가설 : 두 도시에서의 음료 선호도가 동일하지 않다. 
mat_b = np.array([[50, 30, 20],
                  [45, 35, 20]])
mat_b

chi2, p, df, expected = chi2_contingency(mat_b, correction = True)
chi2.round(3) #검정 통계량
p.round(4) #p-value
expected 

#0903 4교시 
#p112 문제
#귀무가설 : 정당 지지와 핸드폰 사용 유무는 독립이다.
#대립가설 : 독립이 아니다.
mat_b = np.array([[49, 47],
                 [15, 27],
                 [32, 30]])
mat_b

chi2, p, df, expected = chi2_contingency(mat_b, correction = True)
chi2.round(3) #검정 통계량
p.round(4) #p-value
expected 

#p105 
from scipy.stats import chisquare
import numpy as np
observed = np.array([13, 23, 24, 20, 27, 18, 15])
expected = np.repeat(20, 7)
statistic, p_value = chisquare(observed, f_exp=expected)
print("Test statistic: ", statistic.round(3))

#지역별 후보 지지율
mat_b = np.array([[176, 124],
                  [193, 107],
                  [159, 141]])
mat_b

chi2, p, df, expected = chi2_contingency(mat_b, correction = True)
chi2.round(3) #검정 통계량
p.round(4) #p-value
#유의수준 0.05보다 p값이 크므로, 귀무가설을 기각할 수 없다. 
expected 