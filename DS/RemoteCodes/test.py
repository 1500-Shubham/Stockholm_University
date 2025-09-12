import numpy as np
import pandas as pd

a = np.asarray([1,2,3])
b = np.zeros((3,4)) #2d
arr = np.ones(3) #1d
# l,c = arr.shape
c= np.full((2,2),10)
ran = np.random.randint(0,10,(3,4),dtype=int) # [0,10] range hai
print(ran)

df = pd.read_csv("data.csv")
# print(df.head())
# print(df.shape)