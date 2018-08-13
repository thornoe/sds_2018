import pandas as pd
import numpy as np

################################################
################## while-loop ##################
################################################
offset = 8
while offset != 0 :
    print("correcting...")
    if offset > 0 :
        offset = offset - 1
    else :
        offset = offset + 1
    print(offset)

################################################
################### for-loop ###################
################################################
areas = [11.25, 18.0, 20.0, 10.75, 9.50]
for area in areas :
    print(area)

# Changing same for-loop to use enumerate()
for index, area in enumerate(areas) :
    print("room " + str(1+index) + ": " + str(area) + " sqm")

# Ex. 2.a. List of lists
house = [["hallway", 11.25],
         ["kitchen", 18.0],
         ["living room", 20.0],
         ["bedroom", 10.75],
         ["bathroom", 9.50]]
for x in house :
    print("the " + x[0] + " is " + str(x[1]) + " sqm")

################################################
################## items-loop ##################
################################################
# Ex. 2.b. Definition of dictionary.
house = {"hallway":11.25,
         "kitchen":18.0,
         "living room":20.0,
         "bedroom":10.75,
         "bathroom":9.50 }
for key, value in house.items() :
    print("the " + key + " is " + str(value) + " sqm")

################################################
################ DataFrame-loop ################
################################################
cars = pd.read_csv('cars.csv', index_col = 0)
# Ex. 3.a
for lab, row in cars.iterrows() :
    print(lab)
    print(row)
# Ex. 3.b
for lab, row in cars.iterrows() :
    print(lab + ": " + str(row['cars_per_cap']))

################################################
################     Apply:     ################
################################################
cars["COUNTRY"] = cars["country"].apply(str.upper)
