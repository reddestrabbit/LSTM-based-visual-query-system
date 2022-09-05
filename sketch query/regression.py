# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 15:03:32 2019

@author: Chaoran Fan
"""

import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import csv
#X = np.array([[1],[2],[3],[4],[5]])

## y = 1 * x_0 + 2 * x_1 + 3
#y = np.dot(X, np.array([3.5])) + 3
#reg = LinearRegression().fit(X, y)
#
#a=reg.coef_
x=[]
y=[]
df = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/dataFitting.csv')
statis=[]
def loadFittingData():
    global x,y
    global statis
    pre_a=2731
    pre_data=2731
    for rowId,row in df.iterrows():
        current_a=int(row['a'])
        if rowId==df.shape[0]-1:
            x.append(float(row['scaleLevel']))
            y.append(float(row['criticalPoints']))
            x=np.array(x)
            y=np.array(y)
            x=x.reshape(len(x),1)
            reg = LinearRegression().fit(x, y)
            tmpStatis=[]
            tmpStatis.append(int(row['dataset']))
            tmpStatis.append(current_a)
            tmpStatis.append(reg.coef_[0])
            tmpStatis.append(reg.score(x,y))
            statis.append(tmpStatis)
            break
        if current_a!=pre_a:   
            print(rowId)
            x=np.array(x)
            y=np.array(y)
            x=x.reshape(len(x),1)
            reg = LinearRegression().fit(x, y)
            tmpStatis=[]
            tmpStatis.append(pre_data)
            tmpStatis.append(pre_a)
            tmpStatis.append(reg.coef_[0])
            tmpStatis.append(reg.score(x,y))
            statis.append(tmpStatis)
            x=[]
            y=[]
            x.append(float(row['scaleLevel']))
            y.append(float(row['criticalPoints']))
            pre_a=current_a
            pre_data=int(row['dataset'])
        else:
            x.append(float(row['scaleLevel']))
            y.append(float(row['criticalPoints']))

#if int(row['dataset'])==1 and int(row['a'])==0:
#def loadFittingData():
#    global x,y
#    global statis
#    pre_x=79.08787541713014
#    pre_y=399.8849836003548
#    pre_a=0
#    pre_data=1
#    for rowId,row in df.iterrows():
#        current_x=float(row['x'])
#        current_y=float(row['y'])
#        if current_x!=pre_x and current_y!=pre_y:      
#            print(current_x)
#            print(current_y)
#            x=np.array(x)
#            y=np.array(y)
#            x=x.reshape(len(x),1)
#            reg = LinearRegression().fit(x, y)
#            tmpStatis=[]
#            tmpStatis.append(pre_x)
#            tmpStatis.append(pre_y)
#            tmpStatis.append(pre_data)
#            tmpStatis.append(pre_a)
#            tmpStatis.append(reg.coef_[0])
#            tmpStatis.append(reg.score(x,y))
#            statis.append(tmpStatis)
#            x=[]
#            y=[]
#            x.append(float(row['scaleLevel']))
#            y.append(float(row['criticalPoints']))
#            pre_x=current_x
#            pre_y=current_y
#            pre_a=int(row['a'])
#            pre_data=int(row['dataset'])
#        else:
#            x.append(float(row['scaleLevel']))
#            y.append(float(row['criticalPoints']))            

loadFittingData()

#a=reg.coef_
with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/statisDataFitting.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    for row in statis:
            writer.writerow(row)
#with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/statisDataFittingByPoint.csv', 'a', newline='') as f:
#    writer = csv.writer(f)
#    for row in statis:
#            writer.writerow(row)