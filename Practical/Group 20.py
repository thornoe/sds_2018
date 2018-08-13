import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


pwd()
os.chdir('C:/Users/thorn/Onedrive/Dokumenter/GitHub/sds_2018') #Change backslashes to forwardslashes

    groups = pd.read_csv("Practical/SDS groups 2018.csv") # leave out "index_col=0" if not already indexed
    name = groups['Name']
    my_row = name == 'Thor Donsby Noe'
    print(groups[my_row])
    my_grp = groups[groups['Group']==20]
    my_grp
    my_grp.to_csv('Practical/My_grp.csv', index = False)
