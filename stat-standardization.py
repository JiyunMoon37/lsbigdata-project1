import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy.stats import uniform

#X~N(3, 7^2)

from scipy.stats import norm

x = norm.ppf(0.25, loc=3, scale=7)

#표준정규분포의 하위 25% 
z = norm.ppf(0.25, loc=0, scale=1) 

3+z*7 #x랑 같다. 


norm.cdf(5, loc=3, scale=7)
norm.cdf(2/7, loc=0, scale=1)

norm.ppf(0.975, loc = 0, scale=1)

z = norm.rvs(loc =0, scale =1, size = 1000)
sns.histplot(z, stat = "density", color = "gray")
plt.show()

zmin = z.min()
zmax = z.max()
z_values = np.linspace(zmin, zmax, 1000)
pdf_values= norm.pdf(z_values, loc=0, scale = 1)
plt.plot(z_values, pdf_values) #선 

plt.show()
#plt.clf()



# k = np.linspace(-5, 5, 100)
# k_value = norm.pdf(k, loc = 3, scale = np.sqrt(2))
# plt.plot(k, k_value )

plt.show()
plt.clf()



#2교시 
x = z*np.sqrt(2) +3
sns.histplot(z, stat = "density", color = "gray")
sns.histplot(x, stat = "density", color = "green")

zmin, zmax = (z.min(), x.min())

z_values = np.linspace(zmin, zmax, 5000)
pdf_values= norm.pdf(z_values, loc=0, scale = 1)
pdf_values2= norm.pdf(z_values, loc=3, scale = np.sqrt(2))
plt.plot(z_values, pdf_values, color = "red") #선 
plt.plot(z_values, pdf_values2, color = "blue") 

plt.show()
plt.clf()

#3교시 
#Q.X~N(5, 3^2)
x = norm.rvs(loc =5, scale =3, size = 1000)
z = (x-3)/5

sns.histplot(z, stat = "density", color = "gray")
#sns.histplot(x, stat = "density", color = "green")

zmin, zmax = (z.min(), z.max())

z_values = np.linspace(zmin, zmax, 100)
pdf_values= norm.pdf(z_values, loc=0, scale = 1)
#pdf_values2= norm.pdf(z_values, loc=5, scale = 3)
plt.plot(z_values, pdf_values, color = "red") #선 
# plt.plot(z_values, pdf_values2, color = "blue") 

plt.show()
plt.clf()

#X~N(5, 3^2)
#1. X에서 표본을 10개 뽑아서 표본 분산 값 계산 
x = norm.rvs(loc =5, scale =3, size = 10)
s = np.std(x, ddof=1) 
s

#2
x = norm.rvs(loc = 5, scale =3, size = 1000)

#표준화
z = (x-5)/s

sns.histplot(z, stat = "density", color = "gray")
zmin, zmax = (z.min(), z.max())

z_values = np.linspace(zmin, zmax, 100)
pdf_values= norm.pdf(z_values, loc=0, scale = 1)
plt.plot(z_values, pdf_values, color = "red") #선 

plt.show()
plt.clf()

#4교시 
#표본 표준편차 나눠도 표준정규분포가 될까? 
x = norm.rvs(loc =5, scale =3, size = 20)
#s_2 = np.var(x, ddof=1)
s = np.std(x, ddof=1) 
s

#2
x = norm.rvs(loc = 5, scale =3, size = 1000)

#표준화
z = (x-5)/s

sns.histplot(z, stat = "density", color = "gray")
zmin, zmax = (z.min(), z.max())

z_values = np.linspace(zmin, zmax, 100)
pdf_values= norm.pdf(z_values, loc=0, scale = 1)
plt.plot(z_values, pdf_values, color = "red") #선 

plt.show()
plt.clf()

#t분포에 대해 알아보자 
#X~t(df) 
from scipy.stats import t 
?t.pdf() 

#자유도가 4인 t분포의 pdf를 그려보세요!
t_values = np.linspace(-4, 4, 100)
pdf_values= t.pdf(t_values, df=30)
plt.plot(t_values, pdf_values, color = "red", linewidth = 2) #선 

plt.show()
#plt.clf()


#표준정규분포 겹치기 
pdf_values= norm.pdf(t_values, loc=0, scale = 1)
plt.plot(t_values, pdf_values, color = "black", linewidth = 2) #선 
plt.show()

#X ~ ?(mu, sigma^2)
#X bar ~ N(mu, sigma^2/n)
#X bar ~= t(x_bar, s^2/n)

x = norm.rvs(loc = 15, scale = 3, size =16, random_state = 42) 
x

x_bar = x.mean()
n = len(x) 
#모분산을 모를 때 : 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)
x_bar - t.ppf(0.975, df=n-1) * np.std(x, ddof=1) / np.sqrt(n)

#모분산을 알 때 : 모평균에 대한 95% 신뢰구간을 구해보자!
x_bar + norm.ppf(0.975, loc = 0, scale =1) * 3 / np.sqrt(n)
x_bar - norm.ppf(0.975, loc = 0, scale =1) * 3 / np.sqrt(n)













