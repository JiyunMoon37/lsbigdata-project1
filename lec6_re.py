import numpy as np

matrix = np.column_stack((np.arange(1,5), 
                          np.arange(12,16))) #row는 행, column은 열 
print("행렬 : \n", matrix)

print("행렬의 크기 :", matrix.shape)

matrix = np.vstack((np.arange(1, 5), np.arange(12, 16)))
matrix

y = np.arange(1,5).reshape([2,2])
print("1부터 4까지의 수로 채운 행렬 y : \n", y)

np.arange(1,7).reshape((2,3))

np.random.seed(2024)
a = np.random.randint(0, 100, 50).reshape((5, -1))
a

np.arange(1,5).reshape((2,2), order = 'C')
np.arange(1,5).reshape((2,2), order = 'F')

mat_a = np.arange(1, 21).reshape((4, 5), order="F")
mat_a

#인덱싱
mat_a[0, 0]
mat_a[1, 1]
mat_a[2, 3]
mat_a[0:2, 3]
mat_a[1:3, 1:4]

mat_a[3,]
mat_a[3, ::2]

mat_b = np.arange(1, 101).reshape((20, -1))
mat_b
mat_b[1:21:2,]
mat_b[1::2, :]

mat_b[[1, 4, 6, 14], ]

x = np.arange(1, 11).reshape((5,2))*2
x
x[[True, True, False, False, True], 0]

mat_b[:,1]
mat_b[:,1].reshape((-1, 1)) 
mat_b[:, (1,)]
mat_b[:, [1]] 
mat_b[:, 1:2]

mat_b[mat_b[:,1] > 50, :]
































