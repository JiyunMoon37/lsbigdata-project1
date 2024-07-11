#soft copy
a = [1, 2, 3]
a

b = a
b

a[1] = 4
a

b

#deep copy
a = [1, 2, 3]
a

b = a[:]
b = a.copy()

a[1] = 4
a
b 

id(a)
id(b)
