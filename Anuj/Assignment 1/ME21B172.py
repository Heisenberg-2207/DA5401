import pandas as pd
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

n = 400     #size of 2D sparse matrix
inflate = n/100     # factor needed while entering data into sparse matrix
img_df = pd.read_csv('Default Dataset.csv')
img_df.drop([i for i in range(len(img_df)) if img_df.loc[i,'X']>100 or img_df.loc[i,'X']<0 or img_df.loc[i,'Y']>100 or img_df.loc[i,'Y']<0], inplace=True)
img = img_df.to_numpy()
image = np.zeros((n,n))
I_m = np.zeros((n,n))
for i in range(len(img)):
    image[n-1-math.floor(inflate*img[i][1])][math.floor(inflate*img[i][0])] = 1

# Contructing a reversal matrix ( 1 reside on the antidiagonal and all other elements are zero )
for i in range(n):
    I_m[i][n-1-i] = 1

# Creating the rotated image
image_180 = np.matmul(I_m,image)
image_90 = np.transpose(image_180)

# Plotting the sparse matrix 
cmap = matplotlib.colors.ListedColormap(['azure', 'royalblue'])
plt.imshow(image_90, interpolation='none',cmap=cmap,extent=[0,1,1,0])
plt.title('Scatter plot of 90$^{\circ}$ clockwise rotated image')
plt.xlabel('Y-axis')
plt.ylabel('X-axis')
plt.show()
plt.imshow(image_180, interpolation='none',cmap=cmap,extent=[0,1,1,0])
plt.title('Scatter plot of 180$^{\circ}$ rotated image')
plt.ylabel('Y-axis')
plt.xlabel('X-axis')
plt.show()