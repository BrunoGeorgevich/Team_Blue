#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 12:02:31 2019

@author: bruno
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, Dense, MaxPool2D, Dropout, Flatten
from keras.models import Sequential
from keras.utils import to_categorical

from random import randint, random
import pandas as pd
import numpy as np
import cv2
import os

def imshow(*imgs):
    i = 0
    for img in imgs:
        cv2.namedWindow(str(i), cv2.WINDOW_KEEPRATIO)
        cv2.imshow(str(i),img)
        i += 1
    while cv2.waitKey(0) != ord('q'):
        pass
    cv2.destroyAllWindows()

#%%

o_list = os.listdir('O')
o_list.sort(key=lambda a: int(a.split(".")[0]))
o_list = list(map(lambda it: f'O/{it}', o_list))
x_list = os.listdir('X')
x_list.sort(key=lambda a: int(a.split(".")[0]))
x_list = list(map(lambda it: f'X/{it}', x_list))

df = pd.DataFrame(columns=['path','name','class'])

for o in o_list:
    df = df.append({'path':o.split('/')[0], 'name':o.split('/')[1], 'class': 'O'}, True)

for x in x_list:
    df = df.append({'path':x.split('/')[0], 'name':x.split('/')[1], 'class': 'X'}, True)

with open("data.csv",'w') as f:
    f.write(df.to_csv(index=False))
#%%
import os
import shutil

df = pd.read_csv('data.csv')

X = df.values[:, :2]
Y = df.values[:, -1]

Y = LabelEncoder().fit_transform(Y)
Y = to_categorical(Y, 2)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

if 'images' not in os.listdir():
    os.mkdir('images')

if 'train' not in os.listdir('images'):
    os.mkdir('images/train')

if 'test' not in os.listdir('images'):
    os.mkdir('images/test')

for path, name in X_train:
    shutil.copy2(f'{path}/{name}',f'images/train/{path}_{name}')

for path, name in X_test:
    shutil.copy2(f'{path}/{name}',f'images/test/{path}_{name}')

#%%

def annotate(path, name):

    annot_body = '''<annotation>
    	<folder>XXXX</folder>
    	<filename>NNNN</filename>
    	<path>images/XXXX/NNNN</path>
    	<source>
    		<database>Unknown</database>
    	</source>
    	<size>
    		<width>WWWW</width>
    		<height>HHHH</height>
    		<depth>MMMM</depth>
    	</size>
    	<segmented>0</segmented>
    	<object>
    		<name>PPPP</name>
    		<pose>Unspecified</pose>
    		<truncated>0</truncated>
    		<difficult>0</difficult>
    		<bndbox>
    			<xmin>AAAA</xmin>
    			<ymin>BBBB</ymin>
    			<xmax>CCCC</xmax>
    			<ymax>DDDD</ymax>
    		</bndbox>
    	</object>
        </annotation>'''

    image = ~cv2.imread(f'{path}/{name}')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cnts,_ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = list(map(lambda cnt: (cnt, cv2.contourArea(cnt)), cnts))
    cnts = list(filter(lambda cnt: cnt[1] > 50, cnts))
    cnts.sort(key=lambda cnt: cnt[1], reverse=True)

    if len(cnts) == 0:
        return

    c = cnts[0][0]

    x,y,w,h = cv2.boundingRect(c)

    image_class = name.split("_")[0]
    width, height, depth = image.shape
    new_body = annot_body.replace('XXXX', path.split('/')[1])    \
                    .replace("NNNN", name)                  \
                    .replace("WWWW", str(width))            \
                    .replace("HHHH", str(height))           \
                    .replace("MMMM", str(depth))            \
                    .replace("PPPP", str(image_class))      \
                    .replace("AAAA", str(x))                \
                    .replace("BBBB", str(y))                \
                    .replace("CCCC", str(x+w))              \
                    .replace("DDDD", str(y+h))

    with open(f"{path}/{name.replace('.png','.xml')}",'w') as f:
        f.write(new_body)

train_list = os.listdir('images/train/')
train_list = list(filter(lambda name: '.png' in name, train_list))
path = 'images/train'

for name in train_list:
    print(name)
    annotate(path, name)

test_list = os.listdir('images/test/')
test_list = list(filter(lambda name: '.png' in name, test_list))
path = 'images/test'

for name in test_list:
    print(name)
    annotate(path, name)
#%%

def randomize_image(image):
    noised = np.zeros_like(image)
    channels = cv2.split(noised)
    
    for channel in channels:
        channel = cv2.randn(channel, 128, 64)
    
    noised = cv2.merge(channels)
    
    foreground = np.zeros_like(image)
    channels = cv2.split(foreground)
    
    for channel in channels:
        channel +=  np.uint8(30 + 200*random())
    
    foreground = cv2.merge(channels)
    
    image[image == 0] = foreground[image == 0]
    image[image == 255] = noised[image == 255]

    return image


train_list = os.listdir('images/train/')
train_list = list(filter(lambda name: '.png' in name, train_list))
path = 'images/train'

for name in train_list:
    image = cv2.imread(f"{path}/{name}")
    image = randomize_image(image)
    cv2.imwrite(f"{path}/{name}",image)

test_list = os.listdir('images/test/')
test_list = list(filter(lambda name: '.png' in name, test_list))
path = 'images/test'

for name in test_list:
    image = cv2.imread(f"{path}/{name}")
    image = randomize_image(image)
    cv2.imwrite(f"{path}/{name}",image)