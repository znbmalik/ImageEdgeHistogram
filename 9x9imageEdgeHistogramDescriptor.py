#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing required libraries
# 1. glob is imported to read images from the directory
# 2. numpy is used to performed certain numerical operation over pixels values such as; operators convolving to detect edges
# 3. pandas is used to plot histogram in conjunction with matplotlib
from PIL import Image, ImageOps
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import glob


# In[ ]:


# Images list is used to store the images
# Labels will store the corresponding titles of the images
Images = [] 
Labels = [] 
for filename in glob.glob('./dataset/*.jpg'): 
    #Loop will iterate over all images in the directory with extension .jpg
    #In each iteration it is loading an image using PIL then converting it a numpy format and
    #adding current image and its label in Images and Labels, respectively
    #Images are cnverted to numpy as we need to perform some mathmatical operations to detect edges
    im=Image.open(filename)
    Images.append(im)
    #Images.append(np.array(im))
    Labels.append(filename)


# In[ ]:


# function to get EHD vector
def findehd(img):
    r,c,m = np.shape(img) # get the shape of image
    if m==3:
        img=ImageOps.grayscale(img)
        
    M = 6*np.floor(r/6) 
    N = 6*np.floor(c/6)
    
    #Making image dim (width and height) divisible completely by 6
    img= np.array(img)
    img = np.resize(img,(int(M),int(N))) 
    
    #as there will be atleast 6*6=36 blocks therefore, considering 37 bins
    # 36 for 6*6 bloack and +1 is to keep mean of allbins
    #and for each block, considering edges across 5 angles
    allbins = np.zeros((37, 5)) # initializing Bins
    p = 1
    L = 0
    for _ in range(6):
        K = 0
        for _ in range(6):
            block = img[K:K+int(M/6), L:L+int(N/6)] # Extracting (M/6,N/6) block
            # sending image to getbins function by dividing it into blocks
            allbins[p,:] = getbins(np.double(block))
            K = K + int(M/6)
            p = p + 1
        L = L + int(N/6)
    
    GlobalBin = np.mean(allbins) # getting global Bin
    allbins[36,:]= np.round(GlobalBin)#storing mean at 37th index
    #as there are 37 bins and foreach bin there are 5 values so intotal it provides 37*5=185 values
    ehd = np.reshape(np.transpose(allbins),[1,185])
    #print(ehd.shape)# there is one row with 185 values
    ehd = ehd[0,-5:]#It is considering last 5 indices which are storing the mean values across all edges
    return ehd


# function for getting Bin values for each block
def getbins(imgb):
    M,N = imgb.shape
    
    M = 3*np.floor(M/3)
    N = 3*np.floor(N/3)
    
    
    imgb = np.resize(imgb,(int(M),int(N))) # Making block dimension divisible by 2
    bins = np.zeros((1,5)) # initialize Bin
    """Operations, define constant"""
    V = np.array([[-1,-2,-1],[0,0,0],[1,2,1]]) # vertical edge operator
    H = np.array([[-1,0,1],[-2,0,2],[-1,0,1]]) # horizontal edge operator
    D45 = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])# diagonal 45 edge operator
    D135 = np.array([[0,1,2],[-1,0,1],[-2,-1,0]]) # diagonal 135 edge operator
    Isot = np.array([[-1,0,1],[0,0,0],[1,0,-1]]) # isotropic edge operator
    T = 50 # threshold
    
    nobr = int(M/3) # loop limits
    nobc = int(N/3) # loop limits
    L = 0

    """loops of operating"""
    for _ in range(nobc):
        K = 0
        for _ in range(nobr):
            block = imgb[K:K+3, L:L+3] # Extracting 3x3 block
            pv = np.abs(np.sum(np.sum(block*V))) # apply operators
            ph = np.abs(np.sum(np.sum(block*H)))
            pd45 = np.abs(np.sum(np.sum(block*D45)))
            pd135 = np.abs(np.sum(np.sum(block*D135)))
            pisot = np.abs(np.sum(np.sum(block*Isot)))
            parray = [pv,ph,pd45,pd135,pisot]
            index = np.argmax(parray) # get the index of max value
            value = parray[index] # get the max value

            if value >= T:
                bins[0,index]=bins[0,index]+1 # update bins values
            K = K+3
        L = L+3

    return bins


# In[ ]:


for i in range(len(Images)):
    display(Images[i],'No. '+str(i+1)+' '+str(Labels[i]).split('\\')[-1])
    ehd_current = findehd(Images[i])
    print("ehd is: " + str(ehd_current))
    
    # Creating histogram and plotting using Pandas
    df=pd.DataFrame({'Orientation':['V','H','D45', 'D135','Isot'], 'Freq':ehd_current})
    df.plot(kind='bar' , x='Orientation', linewidth=1, title='Image Edge Histogram', rot=30, fontsize=13, figsize= (11,6))
    plt.show()


# In[ ]:




