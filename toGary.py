
# coding: utf-8

# In[1]:

from PIL import Image
import csv


# In[2]:

inputPic = open("picId.csv")
inputPicName = []


# In[3]:

for i in csv.reader(inputPic):
    inputPicName.append(i)


# In[4]:

for i in range(len(inputPicName)):
    im = Image.open(inputPicName[i][0])
    img = im.crop((137, 66, 952, 781))
    img = img.convert('L')
    img.save('new_gray/'+str(inputPicName[i][0]))
