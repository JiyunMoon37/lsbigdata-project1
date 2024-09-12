import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-1, 7, 400)
y = np.linspace(-1, 7, 400)
x, y = np.meshgrid(x, y)

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 그래프를 그리기 위한 설정
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 표면 그래프를 그립니다.
ax.plot_surface(x, y, z, cmap='viridis')

# 레이블 및 타이틀 설정
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('f(x, y)')
ax.set_title('Graph of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()

# ==========================
# 등고선 그래프

import numpy as np
import matplotlib.pyplot as plt

# x, y의 값을 정의합니다 (-1에서 7까지)
x = np.linspace(-10, 10, 400)
y = np.linspace(-10, 10, 400)
x, y = np.meshgrid(x, y)

x.shape
y.shape

# 함수 f(x, y)를 계산합니다.
z = (x - 3)**2 + (y - 4)**2 + 3

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(x, y, z, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

#특정 점 (9,2)에 파란색 점을 표시 
plt.scatter(9, 2, color = 'red', s=50)

x = 9; y = 2
lstep = 0.1
for i in range(100) :
    x , y = np.array([x, y]) - lstep * np.array([2*x-6, 2*y-8])
    plt.scatter(float(x), float(y), color = 'red', s = 50)

# 축 레이블 및 타이틀 설정
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Contour Plot of f(x, y) = (x-3)^2 + (y-4)^2 + 3')

# 그래프 표시
plt.show()


##0829 5교시
### 실습: 다음을 최소로 만드는 베타 벡터
'''
f(beta_0, beta_1) = (1 - (beta_0 + beta_1))**2 + 
                    (4 - (beta_0 + 2* beta_1))**2 +
                    (1.5 - (beta_0 + 3* beta_1))**2 +
                    (5 - (beta_0 + 4* beta_1))**2
'''

import numpy as np
import matplotlib.pyplot as plt

beta_0 = np.linspace(-10, 10, 400)
beta_1 = np.linspace(-10, 10, 400)
beta_0, beta_1 = np.meshgrid(x, y)

# 함수 f(beta_0, beta_1)를 계산합니다.
beta = (1 - (beta_0 + beta_1))**2 + \
       (4 - (beta_0 + 2* beta_1))**2 + \
       (1.5 - (beta_0 + 3* beta_1))**2 + \
       (5 - (beta_0 + 4* beta_1))**2

beta_0 = 10; beta_1 = 10
lstep = 0.1
for i in range(1000) :
    beta_0 , beta_1 = np.array([beta_0, beta_1]) \
        - lstep * np.array([-23+8*beta_0+20*beta_1, -67+20*beta_0+60*beta_1])
    plt.scatter(float(beta_0), float(beta_1), color = 'red', s = 50)

# 등고선 그래프를 그립니다.
plt.figure()
cp = plt.contour(beta_0, beta_1, beta, levels=20)  # levels는 등고선의 개수를 조절합니다.
plt.colorbar(cp)  # 등고선 레벨 값에 대한 컬러바를 추가합니다.

# 축 레이블 및 타이틀 설정
plt.xlabel('beta_0')
plt.ylabel('beta_1')
plt.title('beta')

# x축 및 y축 범위 설정
#plt.xlim(-4, 4)
#plt.ylim(-4, 4)

# 그래프 표시
plt.show()

# 최솟값 출력
print(f"최솟값에 도달한 beta_0: {beta_0}, beta_1: {beta_1}")
