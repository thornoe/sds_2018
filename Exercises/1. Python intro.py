# Import packages and set working directory
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pwd()
os.chdir('C:/Users/thorn/Onedrive/Dokumenter/GitHub/sds_2018')  # Change backslashes to forwardslashes

### Data types
A = float(4.11)  # e.g. 3.14, 0.011
B = int(1)  # e.g. 1, 3, 8
A.as_integer_ratio()  # integer ratio ??
C = str('some text')
D = bool()  # True/False
E = list()
F = []
type(F)  # list()
G = np.arange(6).reshape((2,3))
G
G.T  # Transpose (or np.transpose(G))
type(G)

### Validate statements
1 == 5%2  # '%' is the modulus operator, which subtracts integer division
    # output from the input and return the remainder
True & False
True | False
not (True & True)

# Loops
for i in range(1, 6):
    print(i**3)

i = 1
while i < 6:
    print(i**3)
    i += 1

### Containers and arrays
A = [2, 3, 1]
B = [3, 7, 4]
B.index(3)  # where is 3 (first) present in B?

C = [A, B]
type(C)
C_a = np.array(C)
C_a

3 in B
B[1] = 9  # replace middle element of `B` with 9
B[1:3]

max(B)
sorted(B)

len(A + B)

A_s = set(A)
B_s = set(B)
C_s = A_s | B_s  # combine the two sets into one
C_s  # skipping dublicates

np.sum(C_a)
C_a.T  # Transpose
