# Imports
import sys
sys.executable

import os # imports everything from os (folder)
    os.path.exists('randoms.py')

from os import path # only import the function "path" from os

import numpy as np # Name the numpy-matrix setup "np"

#Setting working directory:
pwd()
os.chdir('C:/Users/thorn/Onedrive/Dokumenter/GitHub/Adv.Microeconometrics/Python tutorials') #Change backslashes to forwardslashes

print("See \n this")
#/n is line change

#Integers and floats:
int(1.05)
float(1)

x = 0
for i in range(1000):
    x += 1/3
x

#Loops
List = ['a','b','c']
for l in List:
    print(l)

range(3) == [0,1,2] #Generator, i.e. automatic generating
list(range(3)) == [0,1,2] #List

#Functions
def name(x,y = 7):
    x += 1 # short for x = x+1

    return x+y
name(3)
name(3,3) #Redefining Y
name(3,3,3)


class DOG: #Doesn't need seperate files. Can contain different functions.
    def __init__(self, size): #Fortolker første argument som sig selv
        self.size = size

    def plus_k(self,n):
        return n + self.size

    def plus_n(self,n,m):
        for _ in range(m): # '_' is equivalent to 'i', but is used if we don't use the i's
            n = self.plus_k(n)
        return n

fido = DOG(size = 3) #Fido lægger altid 3 til.

fido.plus_k(1) #Dvs. 1+size
# Self is the reference to "Fido's function" and not just the function of a random function.

DOG.plus_n(3,5,3)
