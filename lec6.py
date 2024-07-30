#lec6 행렬

import numpy as np

#두 개의 벡터를 합쳐 행렬 생성

matrix = np.vstack(
    (np.arange(1, 5),
    np.arange(12, 16))
    )
print("행렬:\n", matrix)
# 행렬의 크기를 재어주는 shape 속성
print("행렬의 크기:", matrix.shape)


np.zeros(5)
np.zeros((5,4))

#채우면서 만들기
np.arange(1, 5).reshape([2, 2]) 
print("1부터 4까지의 수로 채운 행렬 y:\n", y)
 
 
np.arange(1,7).reshape((2, 3))
np.arange(1,7).reshape((2, -1))

#Q. 0에서 99까지 수 중 랜덤하게 50개 숫자를 뽑아서 
#5 by 10 행렬을 만드세요. 

np.random.seed(2024)
a = np.random.randint(0, 100, 50).reshape((5, -1))
a

np.arange(1,21).reshape((4, 5))

#행렬을 채우는 방법 - order
-> order = 'C' : 행우선
-> order = 'F' : 열 우선

mat_a = np.arange(1,21).reshape((4, 5), order = 'F')

#인덱싱
mat_a[0, 0]
mat_a[1, 1]
mat_a[2, 3]
mat_a[0:2, 3]
mat_a[1:3, 1:4]

#행자리, 열자리 비어있는 경우 전체 행, 또는 열 선택
mat_a[3,]
mat_a[3, :]
mat_a[3, ::2]

#짝수행만 선택하려면? 
mat_b = np.arange(1,101).reshape((20, -1))
mat_b
mat_b[1:21:2,] #나
mat_b[1::2, :]

mat_b[[1, 4, 6, 14], ] #1행, 4행, 6행, 14행에 해당하는 열 전체 가져옴.

x = np.arange(1, 11).reshape((5, 2)) * 2
x[[True, True, False, False, True], 0] #2차원이 1차원으로 

mat_b[:, 1] #벡터 
mat_b[:, 1].reshape((-1, 1)) #행렬 
mat_b[:, (1,)] #행렬 (2차원)
mat_b[:, [1]] #행렬 
mat_b[:, 1:2]

#필터링
mat_b[mat_b[:, 1] % 7 ==0, :]

mat_b[:, 1]
mat_b[:, 1] % 7
mat_b[:, 1] % 7 ==0
mat_b[mat_b[:, 1] % 7 ==0, :]

#2교시시
#사진은 행렬이다. 
import numpy as np
import matplotlib.pyplot as plt

#난수 생성하여 3x3 크기의 행렬 생성
np.random.seed(2024)
img1 = np.random.rand(3,3)
print("이미지 행렬 img1:\n", img1)

plt.imshow(img1, cmap = 'gray', interpolation = 'nearest')
plt.colorbar()
plt.show()

a = np.random.randint(0, 256, 20).reshape(4, -1)
a / 255
plt.show 

#6장 강의 p18 
import urllib.request
img_url = "https://bit.ly/3ErnM2Q"
urllib.request.urlretrieve(img_url, "jelly.png")

#!pip install imageio
import imageio

#이미지 읽기 
jelly = imageio.imread("img/jelly.png")
print("이미지 클래스:", type(jelly))
print("이미지 차원:", jelly.shape)
print("이미지 첫 4x4 픽셀, 첫 번째 채널 :\n", jelly[:4, :4, 0])

jelly[:, :, 0].shape
jelly[:, :, 0].transpose().shape

plt.imshow(jelly)
plt.imshow(jelly[:, :, 0].transpose())
plt.imshow(jelly[:, :, 0]) #R
plt.imshow(jelly[:, :, 1]) #G
plt.imshow(jelly[:, :, 2]) #B
plt.imshow(jelly[:, :, 3]) #투명도 아예 없음 
plt.axis('off') #축 정보 없애기 
plt.show()

#p16
#3차원 배열
#두 개의 2x3 행렬 생성
mat1 = np.arange(1, 7).reshape(2, 3)
mat2 = np.arange(7, 13).reshape(2, 3)

my_array = np.array([mat1, mat2])
my_array.shape 

my_array2 = np.array([my_array, my_array])
my_array2
my_array2.shape


first_slice = my_array[0, :, :]
first_slice

first_array = my_array[:, :, :-1]
first_array

my_array[:, :, [0,2]]
my_array[:, :, [0,2]]
my_array [:, 0, :]
my_array[0, 1, 1:3] #1:3 대신 [1,2] 넣어도 된다. 

mat_x = np.arange(1,101).reshape((5, 5, 4))
mat_x
mat_y = np.arange(1,100).reshape((-1, 3, 3))
mat_y

#넘파이 배열 메서드 
a = np.array([[1, 2, 3], [4, 5, 6]])

a.sum()
a.sum(axis = 0)
a.sum(axis = 1)
a

a.mean()
a.mean(axis=0)
a.mean(axis=1)

mat_b = np.random.randint(0, 100, 50).reshape((5, -1))
mat_b

#가장 큰 수는?
mat_b.max()

#행별로 가장 큰 수는?
mat_b.max(axis=1)

#열별 가장 큰 수는?
mat_b.min(axis = 0)

a = np.array([1, 3, 2, 5])
a.cumsum() #행별 누적합 
a.cumprod()
a

mat_b.cumsum(axis=1)
mat_b.sum(axis=1)
mat_b

mat_b.cumprod(axis=1)

mat_b.reshape((2, 5, 5)) #2개, 5행, 5열 
mat_b.flatten()

mat_b.reshape((2, 5, 5)).flatten()

d = np.array([1, 2, 3, 4, 5])
d
d.clip(2, 4)

d.tolist()















