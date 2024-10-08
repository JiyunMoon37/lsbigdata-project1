from scipy.stats import uniform
#uniform에서는 loc이 시작점, scale은 구간 길이 (균일분포)
import numpy as np
import matplotlib.pyplot as plt

uniform.rvs(loc = 2, scale = 4, size = 1)
uniform.pdf(3, loc=2, scale = 4)
uniform.pdf(7, loc=2, scale = 4)

k = np.linspace(0, 8, 100)
y = uniform.pdf(k, loc=2, scale = 4)
plt.plot(k, y, color = "black")
plt.show()
plt.clf()

#예제
uniform.cdf(3.25, loc = 2, scale = 4)

#예제
uniform.cdf(8.39, loc = 2, scale = 4) - uniform.cdf(5, loc = 2, scale = 4)

#상위 7%값은?
uniform.ppf(0.93, loc = 2, scale = 4) 

#표본 20개 뽑아서 표준평균 계산하기
#X ~ 균일분포 U(a, b)

#신뢰구간 
x = uniform.rvs(loc=2, scale=4, size=20*1000, 
                random_state=42)
x=x.reshape(-1, 20)
x.shape
blue_x = x.mean(axis=1)
blue_x

import seaborn as sns

sns.histplot(blue_x, stat = "density")
plt.show()
plt.clf()

#신뢰구간
#X bar ~ N(mu, sigma^2/n)
#X bar ~ N(4, 1.3333333/20)
from scipy.stats import norm

#Plot the normal distribution PDF
x_values = np.linspace(3, 5, 100)
pdf_values = norm.pdf(x_values, loc = 4, 
                    scale = np.sqrt(1.3333333/20))
plt.plot(x_values, pdf_values, color = 'red', linewidth = 2)

#표본평균(파란벽돌) 점찍기 
blue_x = uniform.rvs(loc=2, scale=4, size=20).mean()
#norm.ppf(0.975, loc=0, scale=1) == 1.96
a = blue_x + 1.96 * np.sqrt(1.3333333/20)
b = blue_x - 1.96 * np.sqrt(1.3333333/20)

plt.scatter(blue_x,0.002, 
            color='blue', zorder=10, s = 10)
plt.axvline(x=a, color = 'blue',
            linestyle = '--', linewidth=1)
plt.axvline(x=b, color = 'blue',
            linestyle = '--', linewidth=1)
plt.show()

#기댓값 표현
plt.axvline(x=4, color = 'green', linestyle = '--', linewidth=2)
plt.show()
plt.clf()

norm.ppf(0.025, loc = 4, scale = np.sqrt(1.3333333/20))
norm.ppf(0.975, loc=4, scale= np.sqrt(1.3333333/20))








uniform.pdf(x, loc=0, scale=1)
uniform.cdf(x, loc=0, scale=1)
uniform.ppf(q, loc=0, scale=1)
uniform.rvs(loc=0, scale=1, size=None, random_state=None)
