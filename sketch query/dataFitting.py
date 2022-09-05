# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:37:19 2019

@author: Chaoran Fan
"""
import numpy as np
import csv
from pandas import read_csv
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from time import time
from gensim.models import KeyedVectors
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import datetime
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda
import keras.backend as K
from keras.optimizers import Adadelta
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.layers import merge
import tensorflow as tf
from tensorflow.python.keras.layers import Layer
from keras.models import load_model
import keras
from scipy import interpolate
from scipy.signal import argrelmin, argrelmax
import math
from scipy.interpolate import UnivariateSpline
import copy
canvasWidth1=900
canvasHeight1=400
canvasWidth2=600
canvasHeight2=200
pX=0
py=0
dataset=[]
trace=[]
traceX=[]
traceY=[]
radius=2
Y=[]
initPos1=100
initPos2=500
sel1=False
sel2=False
finished=False
selected=[]
interval=0
length=0
minDataY=0
maxDataY=0
allData=[]
userId=1
count=0
userStudyData=[]
dataLoaded=False
x=[]
y=[]
inflectionPoints=[]
extrema=[]
y1=[]
y2=[]
salientPoints=[]
xSamples=[]
ySamples=[]
cubicSpline=[]
originalPoints=[]
scaled=False
scalePoint=[]
testScale=[1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]
currentDataSetName=""
criticalPointsCount=0

def loadData():
    global dataLoaded
    dataLoaded=True
    
    global currentDataSetName
    global dataset

    dataset = read_csv("C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/realData/"+currentDataSetName+".csv")
    global minDataY,maxDataY
    maxDataY=dataset[dataset.columns[1]].max()
    minDataY=dataset[dataset.columns[1]].min()
    global length
    length=len(dataset)
    global interval
    interval=canvasWidth1*1.0/(length-1)
    print(interval)
 
    global allData
    allData=[]
    #radius=2
    global x,y,y1,y2
    x= np.linspace(0,canvasWidth1,length)
    allData.append(canvasHeight1-(dataset.iloc[0,1]-minDataY)/(maxDataY-minDataY)*canvasHeight1)

    for i in range(1,length):
        allData.append(canvasHeight1-(dataset.iloc[i,1]-minDataY)/(maxDataY-minDataY)*canvasHeight1)
          
    y= np.array(allData)
    global cubicSpline
    cubicSpline= UnivariateSpline(x,y)
    
    global xSamples, ySamples
    xSamples=np.linspace(0,canvasWidth1,canvasWidth1)

    cubicSpline.set_smoothing_factor(0)
    ySamples=cubicSpline(xSamples)
    #print(ySamples)
    global originalPoints
    originalPoints= np.c_[xSamples, ySamples]
    y2=[]
    for i in range(0,len(xSamples)):
        y2.append(cubicSpline.derivatives(xSamples[i])[2])
    compSalientPoints()


def compSalientPoints():
    global inflectionPoints,extrema
    inflectionPoints=[]
    #extrema=[0]
    extrema=[]
    
    min_idxs = argrelmin(ySamples)
    print("min:", len(min_idxs[0]))
    for i in range(0,len(min_idxs[0])):
        extrema.append(min_idxs[0][i])

    max_idxs = argrelmax(ySamples)
    for i in range(0,len(max_idxs[0])):
        extrema.append(max_idxs[0][i])
    print("max:",len(max_idxs[0]))

    if len(xSamples)-1 not in extrema:
        extrema.append(len(xSamples)-1)
    if 0 not in extrema:
        extrema.append(0)
     
    for i in range (1,len(y2)-1):
        if y2[i-1]*y2[i+1]<0:
            y2[i]=0
            inflectionPoints.append(i)
    print("inflectionPoints：",len(inflectionPoints))
    
    global salientPoints
    salientPoints=extrema+inflectionPoints
    salientPoints.sort()
    
    #countCriticalPoints()
    
def smoothData(val):
    print(val)
    global cubicSpline, ySamples
    cubicSpline = UnivariateSpline(x,y)
    cubicSpline.set_smoothing_factor(float(val))
    
    print("smooth:",float(val))
    
    ySamples=cubicSpline(xSamples)
    #print(ySamples)
    global y2
    y2=[]
    for i in range(0,len(xSamples)):
        y2.append(cubicSpline.derivatives(xSamples[i])[2])

    global originalPoints
    originalPoints= np.c_[xSamples, ySamples]

        
    compSalientPoints()    

def countCriticalPoints():
    global criticalPointsCount
    criticalPointsCount=0
    for i in range(0,len(extrema)):
        if originalPoints[extrema[i]][0]>=0 and originalPoints[extrema[i]][0]<=canvasWidth1 and originalPoints[extrema[i]][1]>=0 and originalPoints[extrema[i]][1]<=canvasHeight1:
            criticalPointsCount+=1
            #print("extrema:",extrema[i])
        
    for i in range(0,len(inflectionPoints)):
        if originalPoints[inflectionPoints[i]][0]>=0 and originalPoints[inflectionPoints[i]][0]<=canvasWidth1 and originalPoints[inflectionPoints[i]][1]>=0 and originalPoints[inflectionPoints[i]][1]<=canvasHeight1:
            criticalPointsCount+=1
            #print("inflectionPoints:",inflectionPoints[i])
    print("criticalPoints:",criticalPointsCount)

def generateDataFitting():
    global currentDataSetName
    #k0=[42542, 11626, 350220, 114771, 4277, 479645, 51874, 22281, 494045, 458606]
    #k0=[128422, 14080, 143563, 86975, 6267, 454771, 26831, 57153, 45235, 107590]
    k0=[64218,20166,108728,72811,9054,549923,35271,39221,67734,109528,30065,950309,282598,4854888,65032,556925,1011826,187012]
    #a=[0, 50,100,200,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500]
    #a=[1281,1282,1283,1284,1285,1286,1287,1288,1289]
    a=[2731,2732,2733,2734,2735,2736,2737,2738,2739]
    global originalPoints, scalePoint
    for i in range(12,13):#11
        if i==1:
            currentDataSetName="goldPrice_m"
        if i==2:
            currentDataSetName="weekly-demand-for-a-plastic-cont"
        if i==3:
            currentDataSetName="oil-and-mining"
        if i==4:
            currentDataSetName="monthly-closings-of-the-dowjones"
        if i==5:
            currentDataSetName="annual-common-stock-price-us-187"
        if i==6:
            currentDataSetName="daily-foreign-exchange-rates-31-"
        if i==7:
            currentDataSetName="monthly-boston-armed-robberies-j"
        if i==8:
            currentDataSetName="numbers-on-unemployment-benefits"
        if i==9:
            currentDataSetName="coloured-fox-fur-production-nain"
        if i==10:
            currentDataSetName="chemical-concentration-readings"
        if i==11:
            currentDataSetName="shampoo"      
        if i==12:
            currentDataSetName="daily-total-female-births"
        if i==13:
            currentDataSetName="_imports"        
        if i==14:
            currentDataSetName="ammonia"
        if i==15:
            currentDataSetName="raw-material-height"
        if i==16:
            currentDataSetName="paper-basis-weight"
        if i==17:
            currentDataSetName="rubber-colour"
        if i==18:
            currentDataSetName="_jobs"
        #print(currentDataSetName)
        loadData()
        tmp_originalPoints= copy.deepcopy(originalPoints)
        tmp_salientPoints= copy.deepcopy(salientPoints)
        for l in range(0,len(a)):
            for j in range(1,10):# nine points on the curve to start zoom in
                tmpIndex=tmp_salientPoints[int(len(tmp_salientPoints)*j*0.1)]
                #print("index:",tmpIndex)
                tmp_x=tmp_originalPoints[tmpIndex][0]
                tmp_y=tmp_originalPoints[tmpIndex][1]
                for k in range(0,len(testScale)):
                    if k0[i-1]-a[l]*math.log(testScale[k], 1.004)<0:
                        continue
                    smoothData(k0[i-1]-a[l]*math.log(testScale[k], 1.004))
                    scalePoint=np.array([tmp_x, tmp_y])
                    originalPoints=scalePoint * (1 - testScale[k]) + originalPoints * testScale[k]
                    countCriticalPoints()
                    dataFitting=[]
                    dataFitting.append(tmp_x)
                    dataFitting.append(tmp_y)
                    dataFitting.append(i)
                    dataFitting.append(testScale[k])
                    dataFitting.append(criticalPointsCount)
                    dataFitting.append(k0[i-1])
                    dataFitting.append(a[l])
                    with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/dataFitting.csv', 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(dataFitting)

generateDataFitting()