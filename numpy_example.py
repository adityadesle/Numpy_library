# -*- coding: utf-8 -*-
"""
Created on Sun May 10 12:01:46 2020

@author: 66IN

"""

import numpy as np 
import matplotlib.pyplot as plt


'''
def encrypt(text,s):
    
    ciper=''
    
    for char in text:
        if char == " ":
             ciper = ciper+char
            
        elif char.isupper():
            ciper=ciper+chr((ord(char)+s-65)%26+65)
            
        else:
            ciper = ciper+chr((ord(char)+s-97)%26+97)
            
            
    return ciper            
            
          


text = input("enter the plain text here:")

s = int(input("enter the shift key:"))

print("the original string :",text)
print("the encrypted string",encrypt(text,s))





alphabets = 'abcdefghijklmnopqrstuvwxyz'

str_input=input("enter the string:")

key= int(input("enter the key:"))

n = len(str_input)

str_output=''


for i in range(n):
    character = str_input[i]
    location = alphabets.find(character)
    new_loc=(location+key)%26
    str_output +=alphabets[new_loc]
    


print(str_output)   



#####################################  NUMPY   #########################


a = np.array([1,2,3])

print(a)

b = np.array([[9.0,8.0,7.0],[1.0,2.0,3.0]])

print(b)
print(b[0][1]*b[1][1])

print(a.ndim)  #dimension
print(b.shape,a.dtype) #shape of arrya(row,cols)



a = [1,2,3]

b= [4,5,6]

result = []

for first , second in zip(a,b):
    result.append(first+second)
    


print(result)    

f = np.array([1.2,2.3,3.5])
  
print(f.dtype)

d = np.array([[1,2,3,4,6,5],[4,5,6,7,8,9]])

print(d.ndim)
print(d.shape)


f = np.arange(25).reshape(5,5)

print(f)

red = f[:,1::2]

print(red)



img  = plt.imread('1.png')

plt.imshow(img,cmap=plt.cm.hot)
plt.figure()

plt.show()


a = np.arange(0,80,10)
print(a)
indices = [1,2,-3]

y = a[indices]

print(y)

a[indices] = 99

print(a)


#fancy indexing


a = np.arange(0,25,1).reshape(5,5)
print(a)

mask = np.array([0,0,0,1,0],dtype=bool)

n1=a[mask,1]

n2 = a[[2,3],[3,4]]

n3 = a[[0],[2]]
print(n1)
print(n2)

print(n3)





alphabet = 'abcdefghijklmnopqrstuvwxyz,.'
key      = 'zyxwvutsrqponmlkjihgfrdcba,.'

clear_txt = 'when i arrive leave everyone'


def encrypt():
    
    
    indexVals =[alphabet.index(char.lower()) for char in clear_txt]
    return ''.join(key[indexKey] for indexKey in indexVals)
    


print("original message",clear_txt)

clear = encrypt()
print("encrypted message",clear)



l1 = [1,2,3,4]

l2 = [5,6,7,8]

print(np.add(l1,l2))



x = [1,2,3,4]

print(np.all(x))

print("element with 0 will be discarded")

x  = [0,1,2,3]

print(np.all(x))




#zero or not in a given list


l1 = [1,2,0,4]

count = 0
for i in l1:
    if i==0:
        count +=1
        print("list contain 0 ")
        print(count)
        
    else:
        print("zero not found")        

        
                



l1=[1,0,0,0] 

print(np.all(l1)) 

l2 = [0,0,0]

print(np.all(l2))              
                

#1
a = [12.23,13.32,100,36.32]
array1= np.array(a)

print(array1)

print(type(array1))

#2

a = np.arange(2,11).reshape(3,3)

print(a)

#random values

a1 = np.random.randint(10,size=(3,3))

print(a1)


#3
a = np.zeros(10)

print(a)

a[6]=11

print(a)

#4

a = np.arange(2,38)

print(a)

print(np.flip(a))



#5

a = np.arange(2,10)

print(a)

print(np.asfarray(a))


#6

x = np.arange(10)
 
print(x)
print(x[:5])
print(x[5:])

print(x[4:7])
print(x[::2])

print(x[1::2])

x2 = np.arange(9).reshape(3,3)

print(x2)

print(x2[:3,:2])


#7

x = np.ones([5,5])

print(x)

x[1:-1,1:-1]=0

print(x)

x = np.pad(x, pad_width=1, constant_values=0)
print(x)



#8

l1= [10,20,30]

l2 = [40,50,60,70,80,90]

print(np.concatenate([l1,l2]))


#9

grid =np.array([[1,2,3],[4,5,6]])

print(grid)

print(grid.shape)

print(np.concatenate([grid,grid]))


#10

l1 = [1,2,3,4]

print(l1)

print(type(l1))
print(np.asarray(l1))

print(type(l1))

t1 = (1,2,3,4)

print(t1)

print(type(t1))

print(np.asarray(t1))

print(type(t1))



#11

print (np.full((3,3),6))

print(np.empty((3,4)))


#12
f_value = [0,12,45,76]
print("values in fahrenheit",f_value)
F = np.array(f_value)
print((5*F)/9-(5*32)/9)
print()
 

#13

a = np.array([1,2,3],dtype=float)

print(a.size)

print(a.nbytes)

print(a.itemsize)


      
      

num = [102+23j]

print(np.real(num))

print(np.imag(num))


#14

l1 = np.arange(4)
print(l1)

print("x+5=",l1+5)

print("x**5=",l1**5)



#15
theta = 1/2

print(np.sin(theta))



a = np.array([1,2,3])
b = np.array([5,5,5])

print(a+b)

'''

#fancy indexing
rand = np.random.RandomState()
x = rand.randint(100,size=10)


print(x)

ind = [3,7,9]

print(x[ind])





