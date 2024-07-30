import pandas as pd 
import numpy as np

old_seat = np.arange(1, 29)
new_seat = np.random.choice(old_seat, 28, replace = False)

np.random.seed(20240729)
pd.DataFrame(
    {"old_seat" : old_seat,
    "new_seat" : new_seat}
)

result.to_csv(result, "result.csv")

#y=2x그래프 그리기 
#점을 직선으로 이어서 표현 
import matplotlib.pyplot as plt
from scipy.stats import uniform

x = np.linspace(0, 8, 2)
y = 2*x
plt.scatter(x, y, s=3) 


#y = uniform.pdf(x, loc = 2, scale=4)
plt.plot(x, y, color = "black")
plt.show()
plt.clf() 

#y=x^2를 점 3개를 사용해서 그리기
x = np.linspace(-8, 8, 100)
y = x**2
#plt.scatter(x, y, s=3) 
plt.plot(x, y, color = "black")

#x, y축 범위 설정 
plt.xlim(-10,10)
plt.xlim(0,40) 
plt.gca().set_aspect('equal', adjustable = 'box')

#비율 맞추기
#plt.axis('equal')는 xlim, ylim과 같이 사용 x 

plt.show()
plt.clf() 














