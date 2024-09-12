#0826 2교시 
#y = (x-2)^2 + 1 그래프 그리기
#점을 직선으로 이어서 표현
import matplotlib.pyplot as plt
import numpy as np

x1 = np.linspace(-4, 8, 100)
y1 = (x1 - 2)**2 + 1 
#plt.scatter(x, y, s=3)
plt.plot(x1, y1, color = "black")
plt.xlim(-4, 8)
plt.ylim(0, 15)

#y=4x-11 그래프 
x2 = np.linspace(-4, 8, 100)
y2 = (4 * x2) - 11

plt.plot(x1, y1, color = "black")
plt.plot(x2, y2, color = "red")
plt.xlim(-4, 8)
plt.ylim(0, 15)

k = 4
#f'(x) = 2x-4
#k=4의 기울기
l_slope = 2*k - 4
f_k = (k-2)**2 + 1
l_intercept = f_k - l_slope * k

#y=slop*x + intercept 그래프
line_y = l_slope*x + l_intercept
plt.plot(x, line_y, color = "red")

#0829 3교시
#경사하강법
#y=x^2 경사하강법
#초기값 : 10, 델타 : 0.9
#100번째 x값 
x = 10
lstep = 0.9
for i in range(100) :
    x = x-lstep*(2*x)

print(x)

x = 10
lstep = np.arange(100, 0, -1) * 0.01
len(lstep)

for i in range(100) :
    x = x-lstep[i]*(2*x)

print(x)

#0829 4교시 
#f(x, y) = (x-3)^2 + (y-4)^2 + 3
#시작값 : (9,2), 델타 : 0.1