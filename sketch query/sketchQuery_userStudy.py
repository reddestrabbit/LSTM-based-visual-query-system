# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 16:14:32 2018

@author: Chaoran Fan
"""
import tkinter as tk
from tkinter import ttk
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
from keras.layers import Input, Embedding, LSTM, Lambda, Flatten, TimeDistributed#, Merge
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
import random
from keras.optimizers import Adam, RMSprop

window = tk.Tk()
window.title('my window')
window.geometry('920x480+500+0')
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
var1 = tk.IntVar()
initPos1=100
initPos2=500
sel1=False
sel2=False
finished=False
selected=[]
interval=10
length=0
minDataY=0
maxDataY=0
allData=[]
userId=21
count=0
userStudyData=[]
dataLoaded=False
x=[]
y=[]
inflectionPoints=[]
inflectionPoints_sketch=[]
extrema=[]
extrema_sketch=[]
y1=[]
y2=[]
salientPoints=[]
salientPoints_sketch=[]
x_sketch=[]
y_sketch=[]
xSamples=[]
ySamples=[]
xSamples_sketch=[]
ySamples_sketch=[]
cubicSpline_sketch=[]
cubicSpline=[]
dissimilarity=[]
tmpDissimilarity=[]
widthQ=0
heightQ=0
sketchDelete=True
segments=0
originalPoints=[]
currentX=0
currentY=0
initX=0
initY=0
#cords=[]
currentScale=1.0
scaleAndShiftRecord=[]
cumScale=1
scaled=False
scalePoint=[]
testScale=[1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0]
currentDataSetName=""
criticalPointsCount=0
k0=[64218,20166,108728,72811,9054,549923,35271,39221,67734,109528]
a=[155,107,1371,300,27,1365,250,96,50,1280]
dataNum=0
overAllScale=1.0
scaleCount=0
maxScale=0
timeInterval=1
minIndex=-1
candidates=[]
extra_candidates=[]
curveInSketchPanel_x=[]
curveInSketchPanel_y=[]
originalData=[]
currentSmoothing=0
offsetH=100
sketchPanelColor='#8CF2C2'
leftSketchPanel=350
rightSketchPanel=550
resultMatrix=[]
hunits = 500
N_data=0
N_sketch=0
minIndexOfData=0
indexOfData=0
maxIndexOfData=0
tmpOriginalPoints=[]
tmp_ySamples_sketch=[]
sketchNum=0
sketchRound=0
defaultSketchSmooth=0.1
def loadData(*args):
    global dataLoaded
    global overAllScale, cumScale
    overAllScale=1.0
    cumScale=1.0
    global scaleCount
    scaleCount=0
    global scaleAndShiftRecord
    scaleAndShiftRecord=[]
    dataLoaded=True
    if finished==True:
        btQuery.config(state='normal')
    canvas1.delete("curve")
    canvas1.delete("extrema")
    canvas1.delete("highLight")
    global currentDataSetName
    currentDataSetName=dataName.get()
    global dataNum
    if currentDataSetName=="goldPrice_m":
        dataNum=1
    if currentDataSetName=="weekly-demand-for-a-plastic-cont":
        dataNum=2
    if currentDataSetName=="oil-and-mining":
        dataNum=3
    if currentDataSetName=="monthly-closings-of-the-dowjones":
        dataNum=4
    if currentDataSetName=="annual-common-stock-price-us-187":
        dataNum=5
    if currentDataSetName=="daily-foreign-exchange-rates-31-":
        dataNum=6
    if currentDataSetName=="monthly-boston-armed-robberies-j":
        dataNum=7
    if currentDataSetName=="numbers-on-unemployment-benefits":
        dataNum=8
    if currentDataSetName=="coloured-fox-fur-production-nain":
        dataNum=9
    if currentDataSetName=="chemical-concentration-readings":
        dataNum=10
    
    global dataset
    #with open("C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/"+currentDataSetName+".csv") as csvfile:
    #    dataset = csv.DictReader(csvfile)
    dataset = read_csv("C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/"+currentDataSetName+".csv")
    global minDataY,maxDataY
    maxDataY=dataset[dataset.columns[1]].max()
    minDataY=dataset[dataset.columns[1]].min()
    global length
    length=len(dataset)
    print("length:",length)
    global interval
    interval=canvasWidth1*1.0/(length-1)
    print(interval)
 
    global allData
    allData=[]
    #radius=2
    global x,y,y1,y2
    x= np.linspace(0,canvasWidth1,length)
    allData.append(canvasHeight1-(dataset.iloc[0,1]-minDataY)/(maxDataY-minDataY)*canvasHeight1)
    #canvas1.create_oval(x[0]-radius, allData[0]-radius, x[0]+radius, allData[0]+radius,fill='white', width=1.2, tags='origin')
    for i in range(1,length):
        allData.append(canvasHeight1-(dataset.iloc[i,1]-minDataY)/(maxDataY-minDataY)*canvasHeight1)
        #canvas1.create_line(x[i], canvasHeight1-(dataset.iloc[i,1]-minDataY)/(maxDataY-minDataY)*canvasHeight1, x[i-1], canvasHeight1-(dataset.iloc[i-1,1]-minDataY)/(maxDataY-minDataY)*canvasHeight1, fill='black',tags='curve',joinstyle=tk.ROUND, width=1.2) 
        #canvas1.create_oval(x[i]-radius, allData[i]-radius, x[i]+radius, allData[i]+radius,fill='white', width=1.2, tags='origin')
          
    y= np.array(allData)
    conSum=0
    for i in range(1,len(y)):
        conSum+=abs(y[i]-y[i-1])
    print("conSum:",conSum)
    global cubicSpline
    #cubicSpline= interpolate.splrep(x,y)
    cubicSpline= UnivariateSpline(x,y)
    global xSamples, ySamples
    xSamples=np.linspace(x[0],x[length-1],canvasWidth1+1)
    #print(np.var(y))
    #rangeMin=(length-math.sqrt(2*length))*np.var(y)
    #rangeMax=(length+math.sqrt(2*length))*np.var(y)
    #print(rangeMin)
    #print(rangeMax)
    cubicSpline.set_smoothing_factor(length*np.std(y)*0.03)
    print("smoothing factor",length*np.std(y)*0.03)
    print("std:",np.std(y))
    ySamples=cubicSpline(xSamples)
    #print(ySamples)
    global originalPoints
    originalPoints= np.c_[xSamples, ySamples]
    #global originalData
    #originalData=np.c_[x, y]
    #ySamples=interpolate.splev(xSamples,cubicSpline,0)
#    y1=interpolate.splev(xSamples,cubicSpline,1)
#    y2=interpolate.splev(xSamples,cubicSpline,2)
    y2=[]
    for i in range(0,len(xSamples)):
        y2.append(cubicSpline.derivatives(xSamples[i])[2])
    drawSmoothedData()
    btScale.set(0.03)
    btSketchPanel.config(state='normal')
    btSketchPanel.deselect()
    deleteSketchPanel()
    canvas1.delete("curveInSketchPanel")
    clear()
    #btScale.config(to=1000)

def loadDataForUserStudy():
    global dataLoaded
    global overAllScale
    overAllScale=1.0
    
    dataLoaded=True
    global currentDataSetName
   
    global dataNum
    if dataNum==1:
        currentDataSetName="goldPrice_m" 
    if dataNum==2:
        currentDataSetName="weekly-demand-for-a-plastic-cont"
    if dataNum==3:
        currentDataSetName="oil-and-mining"
    if dataNum==4:
        currentDataSetName="monthly-closings-of-the-dowjones"
    if dataNum==5:
        currentDataSetName="annual-common-stock-price-us-187"
    if dataNum==6:
        currentDataSetName="daily-foreign-exchange-rates-31-"
    if dataNum==7:
        currentDataSetName="monthly-boston-armed-robberies-j"
    if dataNum==8:
        currentDataSetName="numbers-on-unemployment-benefits"
    if dataNum==9:
        currentDataSetName="coloured-fox-fur-production-nain"
    if dataNum==10:
        currentDataSetName="chemical-concentration-readings"
        
    print(currentDataSetName)
    global dataset
    dataset = read_csv("C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/"+currentDataSetName+".csv")
    global minDataY,maxDataY
    maxDataY=dataset[dataset.columns[1]].max()
    minDataY=dataset[dataset.columns[1]].min()
    global length
    length=len(dataset)
    print("length:",length)

    global allData
    allData=[]
    global x,y,y1,y2
    x= np.linspace(0,canvasWidth1,length)
    allData.append(canvasHeight1-(dataset.iloc[0,1]-minDataY)/(maxDataY-minDataY)*canvasHeight1)
    for i in range(1,length):
        allData.append(canvasHeight1-(dataset.iloc[i,1]-minDataY)/(maxDataY-minDataY)*canvasHeight1)
          
    y= np.array(allData)

    global cubicSpline
#    cubicSpline= UnivariateSpline(x,y)
    global xSamples, ySamples
    xSamples=np.linspace(x[0],x[length-1],canvasWidth1+1)

#    cubicSpline.set_smoothing_factor(length*np.std(y)*0.03)
#    print("smoothing factor",length*np.std(y)*0.03)
#    print("std:",np.std(y))
#    ySamples=cubicSpline(xSamples)
#    #print(ySamples)
#    global originalPoints
#    originalPoints= np.c_[xSamples, ySamples]

#    y2=[]
#    for i in range(0,len(xSamples)):
#        y2.append(cubicSpline.derivatives(xSamples[i])[2])
#    drawSmoothedData()



  
def drawSmoothedData():
    canvas1.delete("curve")
    canvas1.delete("extrema")
    canvas1.delete("inflectionPoints")
    global inflectionPoints,extrema
    inflectionPoints=[]
    #extrema=[0]
    extrema=[]
    
#    for i in range(1,len(xSamples)):
#        canvas1.create_line(xSamples[i], ySamples[i], xSamples[i-1], ySamples[i-1], fill='red',tags='curve',joinstyle=tk.ROUND, width=1.5) 
        
#    for i in range(1,originalPoints.shape[0]):
#        canvas1.create_line(originalPoints[i][0], originalPoints[i][1], originalPoints[i-1][0], originalPoints[i-1][1], fill='red',tags='curve',joinstyle=tk.ROUND, width=1.5) 
        
    #canvas1.create_line(xSamples[10], ySamples[10], xSamples[0], ySamples[0], fill='green',tags='curve',joinstyle=tk.ROUND, width=1.5) 
                
    min_idxs = argrelmin(ySamples)
    print(len(min_idxs[0]))
    for i in range(0,len(min_idxs[0])):
        #canvas1.create_oval(xSamples[min_idxs[0][i]]-radius,ySamples[min_idxs[0][i]]-radius,xSamples[min_idxs[0][i]]+radius,ySamples[min_idxs[0][i]]+radius,fill='yellow', width=1.2, tags='extrema')
        extrema.append(min_idxs[0][i])
    #extrema.append(len(xSamples)-1)
    #print(min_idxs)
    max_idxs = argrelmax(ySamples)
    for i in range(0,len(max_idxs[0])):
        #canvas1.create_oval(xSamples[max_idxs[0][i]]-radius,ySamples[max_idxs[0][i]]-radius,xSamples[max_idxs[0][i]]+radius,ySamples[max_idxs[0][i]]+radius,fill='red', width=1.2, tags='extrema')
        extrema.append(max_idxs[0][i])
    print(len(max_idxs[0]))
     
    for i in range (1,len(y2)-1):
        if y2[i-1]*y2[i+1]<0:
            y2[i]=0
            #canvas1.create_oval(xSamples[i]-radius,ySamples[i]-radius,xSamples[i]+radius,ySamples[i]+radius,fill='blue', width=1.2, tags='inflectionPoints')
            inflectionPoints.append(i)
    print(len(inflectionPoints))
    btSliders.config(state='normal')
    
    global salientPoints
    salientPoints=extrema+inflectionPoints
    salientPoints.sort()
    salientPoints=deleteDuplicatedElementFromList(salientPoints)
    
    #countCriticalPoints()
    redraw()


def deleteDuplicatedElementFromList(listA):
    return sorted(set(listA), key = listA.index)

def smoothData(val):
    print("smoothing factor:",val)
    global cubicSpline, ySamples
    cubicSpline = UnivariateSpline(x,y)
    cubicSpline.set_smoothing_factor(length*np.std(y)*float(val))
    
    print("smooth:",length*np.std(y)*float(val))
    
    ySamples=cubicSpline(xSamples)
    #print(ySamples)
    global y2
    y2=[]
    for i in range(0,len(xSamples)):
        y2.append(cubicSpline.derivatives(xSamples[i])[2])
    #ySamples=interpolate.splev(xSamples,cubicSpline,0)
#    y1=interpolate.splev(xSamples,cubicSpline,1)
#    y2=interpolate.splev(xSamples,cubicSpline,2)
    global originalPoints
    originalPoints= np.c_[xSamples, ySamples]
    #print("ddddddd",len(scaleAndShiftRecord))
    for i in range(0,len(scaleAndShiftRecord)):
        if scaleAndShiftRecord[i][2]==0:
            shift=np.array([scaleAndShiftRecord[i][0],scaleAndShiftRecord[i][1]])
            originalPoints+=shift
        else:
            scalePoint_tmp=np.array([scaleAndShiftRecord[i][0],scaleAndShiftRecord[i][1]])
            originalPoints=scalePoint_tmp * (1 - scaleAndShiftRecord[i][2]) + originalPoints * scaleAndShiftRecord[i][2]
        
    drawSmoothedData()
 
def smoothData2(val):
    deselectTheButton()
    if (k0[dataNum-1]-a[dataNum-1]*math.log(overAllScale,1.004))>0:
        smoothData((k0[dataNum-1]-a[dataNum-1]*math.log(overAllScale,1.004)+float(val))/(length*np.std(y)))
    else:
        smoothData((k0[dataNum-1]-a[dataNum-1]*math.log(maxScale,1.004)+float(val))/(length*np.std(y)))

def smoothSketch(val):
    #print(val)
    #print("dddddd")
    if sketchDelete==False:
        global cubicSpline_sketch, ySamples_sketch
        cubicSpline_sketch.set_smoothing_factor(len(x_sketch)*np.std(y_sketch)*float(val))
        #cubicSpline_sketch.set_smoothing_factor(0)
        ySamples_sketch=cubicSpline_sketch(xSamples_sketch)
        global y2_sketch
        y2_sketch=[]
        for i in range(0,len(xSamples_sketch)):
            y2_sketch.append(cubicSpline_sketch.derivatives(xSamples_sketch[i])[2])

        drawSmoothedSketchData()
        
def clear():
    #canvas1.delete("curveInSketchPanel")
    canvas1.delete("sketch")
    canvas1.delete("sketchSplineCurve")
    canvas1.delete("inflectionPoints_sketch")
    canvas1.delete("extrema_sketch")
    
    #canvas2.delete("sketch")
    #canvas2.delete("sketchSplineCurve")
    #canvas2.delete("inflectionPoints_sketch")
    #canvas2.delete("extrema_sketch")
    
    #canvas1.delete("highLight")
#    global sketchNum
#    sketchNum-=1
    global sketchDelete
    sketchDelete=True
    btScale_sketch.set(defaultSketchSmooth)
    global finished
    finished=False
    setButtonDisabled()
    global trace,traceX,traceY
    trace=[]
    traceX=[]
    traceY=[]
    global candidates
    candidates=[]
    
def finishedRate():
    save()
    #rate_btSave.config(state='normal')

global tmp_xSamples, tmp_ySamples
#def save():
#    rate_btSave.config(state='disabled')
#                
#    N=rightSketchPanel-leftSketchPanel+1    
#    
##    global curveInSketchPanel_y, curveInSketchPanel_x
##    curveInSketchPanel_y=np.array(curveInSketchPanel_y)
##    curveInSketchPanel_x=np.array(curveInSketchPanel_x)
#    #tmpSpline= UnivariateSpline(curveInSketchPanel_x,curveInSketchPanel_y)
#    tmpSpline= UnivariateSpline(originalPoints[:,0],originalPoints[:,1])
#    global tmp_xSamples, tmp_ySamples
#    tmp_xSamples=np.linspace(leftSketchPanel,rightSketchPanel,N)
#    tmp_ySamples=tmpSpline(tmp_xSamples)
#    
#    
#    global count
#    count+=1
#    global userStudyData
#    tmpSelected=""
#    tmpTrace=""
#    tmpOriginalTraceX=""
#    tmpOriginalTraceY=""
#
#    for i in range(0,100):
#        tmpSelected=tmpSelected+str(random.random()*canvasHeight1)+";"    
#    for i in range(0,len(tmp_ySamples)):
#        if i==len(tmp_ySamples)-1:
#            tmpSelected=tmpSelected+str(tmp_ySamples[i])
#        else:
#            tmpSelected=tmpSelected+str(tmp_ySamples[i])+";"
#    
#    for i in range(0,len(ySamples_sketch)):
#        if i==len(ySamples_sketch)-1:
#            tmpTrace=tmpTrace+str(ySamples_sketch[i]) 
#        else:
#            tmpTrace=tmpTrace+str(ySamples_sketch[i])+";"
#         
#    if len(ySamples_sketch)<N:
#        tmpTrace=tmpTrace+";"
#        for i in range(0,201-len(ySamples_sketch)):
#            if i==201-len(ySamples_sketch)-1:
#                tmpTrace=tmpTrace+str("-1")
#            else:
#                tmpTrace=tmpTrace+str("-1;")
#    
#    for i in range(0,len(traceX)):
#        if i==len(traceX)-1:
#            tmpOriginalTraceX=tmpOriginalTraceX+str(traceX[i])
#        else:
#            tmpOriginalTraceX=tmpOriginalTraceX+str(traceX[i])+";"
#    
#    for i in range(0,len(trace)):
#        if i==len(trace)-1:
#            tmpOriginalTraceY=tmpOriginalTraceY+str(trace[i])
#        else:
#            tmpOriginalTraceY=tmpOriginalTraceY+str(trace[i])+";"
#            
#    userStudyData=[]
#    userStudyData.append(userId)
#    userStudyData.append(count)
#    userStudyData.append(dataNum)
#    userStudyData.append(tmpSelected)
#    userStudyData.append(tmpTrace)
#    userStudyData.append(tmpOriginalTraceX)
#    userStudyData.append(tmpOriginalTraceY)
#    if var3.get()=='A':
#        rate_bt0.deselect()
#        userStudyData.append(0)
#    if var3.get()=='B':
#        rate_bt025.deselect()
#        userStudyData.append(0.25)
#    if var3.get()=='C':
#        rate_bt050.deselect()
#        userStudyData.append(0.5)
#    if var3.get()=='D':
#        rate_bt075.deselect()
#        userStudyData.append(0.75)
#    if var3.get()=='E':
#        rate_bt1.deselect()
#        userStudyData.append(1.0)
#
#    with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy.csv', 'a', newline='') as f:
#        writer = csv.writer(f)
#        writer.writerow(userStudyData)
#    print("saved")
def save():
    rate_btSave.config(state='disabled')
    #global sketchNum
    global trace,traceX,traceY
    global sketchRound
    global count
    count+=1
    var5.set(str(count))
    global userStudyData
    tmpSelected=""
    tmpTrace="*"
    tmpOriginalTraceX=""
    tmpOriginalTraceY=""

    N=rightSketchPanel-leftSketchPanel+1
    for i in range(0,100):
        tmpSelected=tmpSelected+str(random.random()*canvasHeight1)+";"   
    for i in range(0,100):
        print(random.random()*canvasHeight1)
    for i in range(0,N):
        if i==N-1:
            tmpSelected=tmpSelected+str(tmpOriginalPoints[i+indexOfData][1])
        else:
            tmpSelected=tmpSelected+str(tmpOriginalPoints[i+indexOfData][1])+";"
    
    #print(tmp_ySamples_sketch)         
    for i in range(0,len(tmp_ySamples_sketch)):
        if i==len(tmp_ySamples_sketch)-1:
            tmpTrace=tmpTrace+str(tmp_ySamples_sketch[i])
        else:
            tmpTrace=tmpTrace+str(tmp_ySamples_sketch[i])+";"
    
    for i in range(0,len(traceX)):
        if i==len(traceX)-1:
            tmpOriginalTraceX=tmpOriginalTraceX+str(traceX[i])
        else:
            tmpOriginalTraceX=tmpOriginalTraceX+str(traceX[i])+";"
    
    for i in range(0,len(trace)):
        if i==len(trace)-1:
            tmpOriginalTraceY=tmpOriginalTraceY+str(trace[i])
        else:
            tmpOriginalTraceY=tmpOriginalTraceY+str(trace[i])+";"
            
    userStudyData=[]
    userStudyData.append(userId)
    userStudyData.append(count)
    userStudyData.append(dataNum)
    if sketchRound==5:
        userStudyData.append(extra_candidates[count-1][1])
    else:
        userStudyData.append(candidates[randomShown[count-1]][1])
    userStudyData.append(sketchNum)
    userStudyData.append(tmpSelected)
    userStudyData.append(tmpTrace)
    userStudyData.append(tmpOriginalTraceX)
    userStudyData.append(tmpOriginalTraceY)
    if var3.get()=='A':
        rate_bt0.deselect()
        userStudyData.append(0)
    if var3.get()=='B':
        rate_bt025.deselect()
        userStudyData.append(0.25)
    if var3.get()=='C':
        rate_bt050.deselect()
        userStudyData.append(0.5)
    if var3.get()=='D':
        rate_bt075.deselect()
        userStudyData.append(0.75)
    if var3.get()=='E':
        rate_bt1.deselect()
        userStudyData.append(1.0)
    if sketchRound==5:
        userStudyData.append(extra_candidates[count-1][6])
        userStudyData.append(extra_candidates[count-1][7])
        userStudyData.append(extra_candidates[count-1][8])
        userStudyData.append(extra_candidates[count-1][9])
        userStudyData.append(extra_candidates[count-1][10])
    else:
        userStudyData.append(candidates[randomShown[count-1]][6])
        userStudyData.append(candidates[randomShown[count-1]][7])
        userStudyData.append(candidates[randomShown[count-1]][8])
        userStudyData.append(candidates[randomShown[count-1]][9])
        userStudyData.append(candidates[randomShown[count-1]][10])
    with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(userStudyData)
    
    print("saved")
    
    if sketchRound==5:
        if count<10:
            trace=[]
            traceX=[]
            traceY=[]
            canvas1.delete("sketch")
            canvas1.delete("sketchSplineCurve")
            canvas1.delete("highLight")
            canvas1.delete("inflectionPoints_sketch")
            canvas1.delete("extrema_sketch")
            extraUserStudy()
    else:
        if count<candidatesNum:
            highlightMatchingResultFoUserStudy()
        else:
            sketchRound+=1
            count=0
            trace=[]
            traceX=[]
            traceY=[]
            canvas1.delete("sketch")
            canvas1.delete("sketchSplineCurve")
            canvas1.delete("highLight")
            canvas1.delete("inflectionPoints_sketch")
            canvas1.delete("extrema_sketch")
            var5.set(str(""))
            if sketchRound==5:
                extraUserStudy()
            
        #sketchNum+=1

def sliders():
    if var1.get()==1:
        #initPos1=100
        #initPos2=500
        if finished==True:
            btSave.config(state='normal')
        canvas1.create_line(initPos1, 0, initPos1, canvasHeight1, fill='red',tags='slide1',joinstyle=tk.ROUND, width=1.5) 
        canvas1.create_line(initPos2, 0, initPos2, canvasHeight1, fill='green',tags='slide2',joinstyle=tk.ROUND, width=1.5) 
        print("sliders")
    else:
        canvas1.delete("slide1")
        canvas1.delete("slide2")
        btSave.config(state='disabled')
        print("none")

X_query=[]


queryResults=[]
maxIndex=0
def qetch():
    matchingQetch()
    

#def highlightMatchingResult():
#    canvas1.delete("highLight")
#    if var2.get()=='A':
#        minIndexOfData=dissimilarity.index(min(dissimilarity))
#        print("dissimilarity:",min(dissimilarity))
#        var.set(str('%.2f' % min(dissimilarity)))
#        
#        shift=np.array([traceX[0]-tmp_xSamples[minIndexOfData],traceY[0]-tmp_ySamples[minIndexOfData]])
#        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#        tmpOriginalPoints+=shift
#        for i in range(1,tmpOriginalPoints.shape[0]):
#            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#        for i in range(1,N_sketch+1):
#            canvas1.create_line(tmpOriginalPoints[i+minIndexOfData][0], tmpOriginalPoints[i+minIndexOfData][1], tmpOriginalPoints[i-1+minIndexOfData][0], tmpOriginalPoints[i-1+minIndexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#            
#    if var2.get()=='B':
#        #canvas1.delete("highLight")
#        indexOfData=dissimilarity.index(tmpDissimilarity[int(len(tmpDissimilarity)*0.25)])
#        var.set(str('%.2f' % tmpDissimilarity[int(len(tmpDissimilarity)*0.25)]))
#        shift=np.array([traceX[0]-tmp_xSamples[indexOfData],traceY[0]-tmp_ySamples[indexOfData]])
#        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#        tmpOriginalPoints+=shift
#        for i in range(1,tmpOriginalPoints.shape[0]):
#            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#        for i in range(1,N_sketch+1):
#            canvas1.create_line(tmpOriginalPoints[i+indexOfData][0], tmpOriginalPoints[i+indexOfData][1], tmpOriginalPoints[i-1+indexOfData][0], tmpOriginalPoints[i-1+indexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#         
#    if var2.get()=='C':
#        #canvas1.delete("highLight")
#        indexOfData=dissimilarity.index(tmpDissimilarity[int(len(tmpDissimilarity)*0.5)])
#        var.set(str('%.2f' % tmpDissimilarity[int(len(tmpDissimilarity)*0.5)]))
#        shift=np.array([traceX[0]-tmp_xSamples[indexOfData],traceY[0]-tmp_ySamples[indexOfData]])
#        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#        tmpOriginalPoints+=shift
#        for i in range(1,tmpOriginalPoints.shape[0]):
#            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#        for i in range(1,N_sketch+1):
#            canvas1.create_line(tmpOriginalPoints[i+indexOfData][0], tmpOriginalPoints[i+indexOfData][1], tmpOriginalPoints[i-1+indexOfData][0], tmpOriginalPoints[i-1+indexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#        
#    if var2.get()=='D':
#        #canvas1.delete("highLight")
#        indexOfData=dissimilarity.index(tmpDissimilarity[int(len(tmpDissimilarity)*0.75)])
#        var.set(str('%.2f' % tmpDissimilarity[int(len(tmpDissimilarity)*0.75)]))
#        shift=np.array([traceX[0]-tmp_xSamples[indexOfData],traceY[0]-tmp_ySamples[indexOfData]])
#        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#        tmpOriginalPoints+=shift
#        for i in range(1,tmpOriginalPoints.shape[0]):
#            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#        for i in range(1,N_sketch+1):
#            canvas1.create_line(tmpOriginalPoints[i+indexOfData][0], tmpOriginalPoints[i+indexOfData][1], tmpOriginalPoints[i-1+indexOfData][0], tmpOriginalPoints[i-1+indexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#        
#    if var2.get()=='E':
#        #canvas1.delete("highLight")
#        maxIndexOfData=dissimilarity.index(max(dissimilarity))
#        var.set(str('%.2f' % max(dissimilarity)))
#        shift=np.array([traceX[0]-tmp_xSamples[maxIndexOfData],traceY[0]-tmp_ySamples[maxIndexOfData]])
#        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#        tmpOriginalPoints+=shift
#        for i in range(1,tmpOriginalPoints.shape[0]):
#            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#        for i in range(1,N_sketch+1):
#            canvas1.create_line(tmpOriginalPoints[i+maxIndexOfData][0], tmpOriginalPoints[i+maxIndexOfData][1], tmpOriginalPoints[i-1+maxIndexOfData][0], tmpOriginalPoints[i-1+maxIndexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 

def extraUserStudy():
    #canvas1.delete("highLight")
    global dataNum, indexOfData, tmpOriginalPoints,tmp_xSamples,tmp_ySamples
    indexOfData=extra_candidates[count][0]
    dataNum=extra_candidates[count][5]
    loadDataForUserStudy()
        
    scaleForUserStudy(extra_candidates[count][1])
    tmpOriginalPoints=originalPoints
    widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
    #global N_data, tmp_xSamples, tmp_ySamples
    N_data=int(widthData)
    tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
    tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
    tmp_ySamples=tmpSpline(tmp_xSamples)
        
    
    shift=np.array([extra_candidates[count][3],canvasHeight1/2-(extra_candidates[count][6]+extra_candidates[count][7])/2.0])
    tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
    tmpOriginalPoints+=shift
        
    for i in range(1,tmpOriginalPoints.shape[0]):
        canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND,capstyle=tk.ROUND,width=5.5) 
    
    for i in range(1,rightSketchPanel-leftSketchPanel+1):
        canvas1.create_line(tmpOriginalPoints[i+indexOfData][0], tmpOriginalPoints[i+indexOfData][1], tmpOriginalPoints[i-1+indexOfData][0], tmpOriginalPoints[i-1+indexOfData][1], fill='orange',tags='highLight',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 

def highlightMatchingResultFoUserStudy():
    canvas1.delete("highLight")
    global dataNum, indexOfData, tmpOriginalPoints,tmp_xSamples,tmp_ySamples
    indexOfData=candidates[randomShown[count]][0]
    indexOfData2=candidates[randomShown[count]][9]
    dataNum=candidates[randomShown[count]][5]
    loadDataForUserStudy()
        
    scaleForUserStudy(candidates[randomShown[count]][1])
    tmpOriginalPoints=originalPoints
    widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
    #global N_data, tmp_xSamples, tmp_ySamples
    N_data=int(widthData)
    tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
    tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
    tmp_ySamples=tmpSpline(tmp_xSamples)
        
    shift=np.array([candidates[randomShown[count]][3],candidates[randomShown[count]][4]])
    tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
    tmpOriginalPoints+=shift
        
    for i in range(1,tmpOriginalPoints.shape[0]):
        canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
    
    for i in range(1,N_sketch+1):
        canvas1.create_line(tmpOriginalPoints[i+indexOfData2][0], tmpOriginalPoints[i+indexOfData2][1], tmpOriginalPoints[i-1+indexOfData2][0], tmpOriginalPoints[i-1+indexOfData2][1], fill='orange',tags='highLight',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
            
    
#def highlightMatchingResult():
#    canvas1.delete("highLight")
#    global minIndexOfData,indexOfData,maxIndexOfData,tmpOriginalPoints,originalPoints,traceX,traceY, dataNum
#    if var2.get()=='A':
#        minIndexOfData=dissimilarity[0][0]
#        var.set(str('%.2f' % dissimilarity[0][2]))
#        dataNum=dissimilarity[0][5]
#        loadDataForUserStudy()
#        
#        scaleForUserStudy(dissimilarity[0][1])
#        tmpOriginalPoints=originalPoints
#        widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
#        #global N_data, tmp_xSamples, tmp_ySamples
#        N_data=int(widthData)
#        tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
#        tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
#        tmp_ySamples=tmpSpline(tmp_xSamples)
#        
#        shift=np.array([dissimilarity[0][3],dissimilarity[0][4]])
#        #shift=np.array([dissimilarity[0][3]-tmp_xSamples[minIndexOfData],dissimilarity[0][4]-tmp_ySamples[minIndexOfData]])
#        #shift=np.array([traceX[0]-tmp_xSamples[minIndexOfData],traceY[0]-tmp_ySamples[minIndexOfData]])
#        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#        tmpOriginalPoints+=shift
#        
#        for i in range(1,tmpOriginalPoints.shape[0]):
#            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
##        for i in range(1,N_sketch+1):
##            canvas1.create_line(tmpOriginalPoints[i+minIndexOfData][0], tmpOriginalPoints[i+minIndexOfData][1], tmpOriginalPoints[i-1+minIndexOfData][0], tmpOriginalPoints[i-1+minIndexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#            
#    if var2.get()=='B':
#        #canvas1.delete("highLight")
#        indexOfData=dissimilarity[int(len(dissimilarity)*0.25)][0]
#        var.set(str('%.2f' % dissimilarity[int(len(dissimilarity)*0.25)][2]))
#        
#        dataNum=dissimilarity[int(len(dissimilarity)*0.25)][5]
#        loadDataForUserStudy()
#        
#        scaleForUserStudy(dissimilarity[int(len(dissimilarity)*0.25)][1])
#        tmpOriginalPoints=originalPoints
#        widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
#        #global N_data, tmp_xSamples, tmp_ySamples
#        N_data=int(widthData)
#        tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
#        tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
#        tmp_ySamples=tmpSpline(tmp_xSamples)
#        
#        shift=np.array([dissimilarity[int(len(dissimilarity)*0.25)][3],dissimilarity[int(len(dissimilarity)*0.25)][4]])
#        #shift=np.array([dissimilarity[int(len(dissimilarity)*0.25)][3]-tmp_xSamples[indexOfData],dissimilarity[int(len(dissimilarity)*0.25)][4]-tmp_ySamples[indexOfData]])
#        #shift=np.array([traceX[0]-tmp_xSamples[indexOfData],traceY[0]-tmp_ySamples[indexOfData]])
#        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#        tmpOriginalPoints+=shift
#        for i in range(1,tmpOriginalPoints.shape[0]):
#            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
##        for i in range(1,N_sketch+1):
##            canvas1.create_line(tmpOriginalPoints[i+indexOfData][0], tmpOriginalPoints[i+indexOfData][1], tmpOriginalPoints[i-1+indexOfData][0], tmpOriginalPoints[i-1+indexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#         
#    if var2.get()=='C':
#        #canvas1.delete("highLight")
#        indexOfData=dissimilarity[int(len(dissimilarity)*0.5)][0]
#        var.set(str('%.2f' % dissimilarity[int(len(dissimilarity)*0.5)][2]))
#        
#        dataNum=dissimilarity[int(len(dissimilarity)*0.5)][5]
#        loadDataForUserStudy()
#        
#        scaleForUserStudy(dissimilarity[int(len(dissimilarity)*0.5)][1])
#        tmpOriginalPoints=originalPoints
#        widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
#        #global N_data, tmp_xSamples, tmp_ySamples
#        N_data=int(widthData)
#        tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
#        tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
#        tmp_ySamples=tmpSpline(tmp_xSamples)
#        
#        shift=np.array([dissimilarity[int(len(dissimilarity)*0.5)][3],dissimilarity[int(len(dissimilarity)*0.5)][4]])
#        #shift=np.array([dissimilarity[int(len(dissimilarity)*0.5)][3]-tmp_xSamples[indexOfData],dissimilarity[int(len(dissimilarity)*0.5)][4]-tmp_ySamples[indexOfData]])
#        #shift=np.array([traceX[0]-tmp_xSamples[indexOfData],traceY[0]-tmp_ySamples[indexOfData]])
#        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#        tmpOriginalPoints+=shift
#        for i in range(1,tmpOriginalPoints.shape[0]):
#            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
##        for i in range(1,N_sketch+1):
##            canvas1.create_line(tmpOriginalPoints[i+indexOfData][0], tmpOriginalPoints[i+indexOfData][1], tmpOriginalPoints[i-1+indexOfData][0], tmpOriginalPoints[i-1+indexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#        
#    if var2.get()=='D':
#        #canvas1.delete("highLight")
#        indexOfData=dissimilarity[int(len(dissimilarity)*0.75)][0]
#        var.set(str('%.2f' % dissimilarity[int(len(dissimilarity)*0.75)][2]))
#        
#        dataNum=dissimilarity[int(len(dissimilarity)*0.75)][5]
#        loadDataForUserStudy()
#        
#        scaleForUserStudy(dissimilarity[int(len(dissimilarity)*0.75)][1])
#        tmpOriginalPoints=originalPoints
#        widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
#        #global N_data, tmp_xSamples, tmp_ySamples
#        N_data=int(widthData)
#        tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
#        tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
#        tmp_ySamples=tmpSpline(tmp_xSamples)
#        
#        shift=np.array([dissimilarity[int(len(dissimilarity)*0.75)][3],dissimilarity[int(len(dissimilarity)*0.75)][4]])
#        #shift=np.array([dissimilarity[int(len(dissimilarity)*0.75)][3]-tmp_xSamples[indexOfData],dissimilarity[int(len(dissimilarity)*0.75)][4]-tmp_ySamples[indexOfData]])
#        #shift=np.array([traceX[0]-tmp_xSamples[indexOfData],traceY[0]-tmp_ySamples[indexOfData]])
#        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#        tmpOriginalPoints+=shift
#        for i in range(1,tmpOriginalPoints.shape[0]):
#            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
##        for i in range(1,N_sketch+1):
##            canvas1.create_line(tmpOriginalPoints[i+indexOfData][0], tmpOriginalPoints[i+indexOfData][1], tmpOriginalPoints[i-1+indexOfData][0], tmpOriginalPoints[i-1+indexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#        
#    if var2.get()=='E':
#        #canvas1.delete("highLight")
#        maxIndexOfData=dissimilarity[len(dissimilarity)-1][0]
#        var.set(str('%.2f' % dissimilarity[len(dissimilarity)-1][2]))
#        
#        dataNum=dissimilarity[len(dissimilarity)-1][5]
#        loadDataForUserStudy()
#        
#        scaleForUserStudy(dissimilarity[len(dissimilarity)-1][1])
#        tmpOriginalPoints=originalPoints
#        widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
#        #global N_data, tmp_xSamples, tmp_ySamples
#        N_data=int(widthData)
#        tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
#        tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
#        tmp_ySamples=tmpSpline(tmp_xSamples)
#        
#        shift=np.array([dissimilarity[len(dissimilarity)-1][3],dissimilarity[len(dissimilarity)-1][4]])
#        #shift=np.array([dissimilarity[len(dissimilarity)-1][3]-tmp_xSamples[maxIndexOfData],dissimilarity[len(dissimilarity)-1][4]-tmp_ySamples[maxIndexOfData]])
#        #shift=np.array([traceX[0]-tmp_xSamples[maxIndexOfData],traceY[0]-tmp_ySamples[maxIndexOfData]])
#        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#        tmpOriginalPoints+=shift
#        for i in range(1,tmpOriginalPoints.shape[0]):
#            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
#        for i in range(1,N_sketch+1):
#            canvas1.create_line(tmpOriginalPoints[i+maxIndexOfData][0], tmpOriginalPoints[i+maxIndexOfData][1], tmpOriginalPoints[i-1+maxIndexOfData][0], tmpOriginalPoints[i-1+maxIndexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 

def scaleForUserStudy(val):
    global overAllScale
    scaleStep=1.05
    print("val:",val)
    overAllScale=math.pow(scaleStep,val)
    
    global originalPoints
    global currentSmoothing
    if (k0[dataNum-1]-a[dataNum-1]*math.log(overAllScale,1.004))>0:                       
        currentSmoothing=(k0[dataNum-1]-a[dataNum-1]*math.log(overAllScale,1.004))/(length*np.std(y))        
        cubicSpline = UnivariateSpline(x,y)
        cubicSpline.set_smoothing_factor(length*np.std(y)*currentSmoothing)
         
        ySamples=cubicSpline(xSamples)
        
        originalPoints= np.c_[xSamples, ySamples]
        scalePoint_tmp=np.array([canvasWidth1/2,canvasHeight1/2])
        originalPoints=scalePoint_tmp * (1 - overAllScale) + originalPoints * overAllScale
    else:
        print("only scale without changing smoothing")
        cubicSpline = UnivariateSpline(x,y)
        currentSmoothing=0
        cubicSpline.set_smoothing_factor(length*np.std(y)*currentSmoothing)
        ySamples=cubicSpline(xSamples)
        
        originalPoints= np.c_[xSamples, ySamples]
        scalePoint_tmp=np.array([canvasWidth1/2,canvasHeight1/2])        
        originalPoints=scalePoint_tmp * (1 - overAllScale) + originalPoints * overAllScale
           
    print("overAllScale=",overAllScale)

scaleForData=[[1,6,12,18,24],[3,6,9,12,15],[4,8,12,16,20],[4,8,12,16,20],[5,10,15,20,25],[1,6,12,18,24],[1,6,12,18,24],[3,6,9,12,15],[3,6,9,12,15],[1,3,6,9,12]]
def takeLast(elem):
    return elem[2]

candidatesNum=20
likert=5
randomShown=[]

def sketchInfoComp():
    tmp_x_sketch=np.array(traceX,dtype=float)
    tmp_y_sketch=np.array(traceY,dtype=float)
    
    global N_sketch
    N_sketch=int((traceX[len(traceX)-1]-traceX[0]))
    tmp_cubicSpline_sketch= UnivariateSpline(tmp_x_sketch,tmp_y_sketch)
    tmp_cubicSpline_sketch.set_smoothing_factor(len(x_sketch)*np.std(y_sketch)*defaultSketchSmooth)
    
    global tmp_ySamples_sketch
    tmp_xSamples_sketch=np.linspace(traceX[0],traceX[len(traceX)-1], N_sketch+1)
    tmp_ySamples_sketch= cubicSpline_sketch(tmp_xSamples_sketch)
    tmpLeft=[]
    tmpRight=[]
    for i in range(0,traceX[0]-leftSketchPanel):
        tmpLeft.append(-1)
    for i in range(0,rightSketchPanel-traceX[len(traceX)-1]):
        tmpRight.append(-1)
    
    tmp_ySamples_sketch=np.append(tmpLeft,tmp_ySamples_sketch)
    tmp_ySamples_sketch=np.append(tmp_ySamples_sketch,tmpRight)

def candidatesForUserStudyComp():
    global dissimilarity
    dissimilarity=[]
    #print("tmp_ySamples_sketch",len(tmp_ySamples_sketch))
    global dataNum
    for k in range(0,10):
        dataNum=k+1
        loadDataForUserStudy()
        
        for m in range(0,5):
            scaleForUserStudy(scaleForData[dataNum-1][m])
            tmpOriginalPoints=originalPoints
            widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
            global N_data, tmp_xSamples, tmp_ySamples
            N_data=int(widthData)
            tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
            tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
            tmp_ySamples=tmpSpline(tmp_xSamples)
            
            extrema=[]
            min_idxs = argrelmin(tmp_ySamples)
            #print(len(min_idxs[0]))
            for i in range(0,len(min_idxs[0])):
                extrema.append(min_idxs[0][i])
            max_idxs = argrelmax(tmp_ySamples)
            for i in range(0,len(max_idxs[0])):
                extrema.append(max_idxs[0][i])
            #print(len(max_idxs[0]))
            if len(tmp_xSamples)-1 not in extrema:
                extrema.append(len(tmp_xSamples)-1)
            if 0 not in extrema:
                extrema.append(0)
    
            for i in range(0,N_data+1-(rightSketchPanel-leftSketchPanel+1)+1,20):
                error=0
                flag=0
                tmpExtrema=[]
                tmpClip=[]
                for j in range(0,len(tmp_ySamples_sketch)):
                    tmpClip.append(tmp_ySamples[i+j])
                    if i+j in extrema:
                        tmpExtrema.append(i+j)
                    if tmp_ySamples_sketch[j]==-1:
                        continue
                    else:
                        if flag==0:
                            #shift=(max(traceY)+min(traceY))/2-(max(tmp_ySamples[i+j:i+j+N_sketch+1])+min(tmp_ySamples[i+j:i+j+N_sketch+1]))/2
                            #shift=tmp_ySamples_sketch[j]-tmp_ySamples[i+j]
                            shift=np.mean(traceY)-np.mean(tmp_ySamples[i+j:i+j+N_sketch+1])
                            shiftX=leftSketchPanel-tmp_xSamples[i]
                            #shiftY=tmp_ySamples_sketch[j]-tmp_ySamples[i+j]
                            #shiftY=(max(traceY)+min(traceY))/2-(max(tmp_ySamples[i+j:i+j+N_sketch+1])+min(tmp_ySamples[i+j:i+j+N_sketch+1]))/2
                            shiftY=np.mean(traceY)-np.mean(tmp_ySamples[i+j:i+j+N_sketch+1])
                            comparePoint=i+j
                            flag=1
                    error+=np.square(tmp_ySamples[i+j]+shift-tmp_ySamples_sketch[j])
            #dissimilarity.append(np.sqrt(error)/len(tmp_ySamples_sketch))

                dissimilarity.append([i,scaleForData[dataNum-1][m],np.sqrt(error)/len(tmp_ySamples_sketch),shiftX,shiftY,dataNum,max(tmpClip),min(tmpClip),len(tmpExtrema),comparePoint])
            #dissimilarity.append([i,overAllScale,np.sqrt(error)/len(tmp_ySamples_sketch)])
      #(max(tmp_ySamples[117:236])+min(tmp_ySamples[117:236]))/2      
#        for i in range(0,N_data+1-(N_sketch+1)+1):
#            error=0
#            shift=tmp_ySamples[i]-tmp_ySamples_sketch[0]
#            
#            for j in range(0,len(tmp_ySamples_sketch)):
#                error+=np.square(tmp_ySamples_sketch[j]-tmp_ySamples[i+j]+shift)
#            #dissimilarity.append(np.sqrt(error)/len(tmp_ySamples_sketch))
#            dissimilarity.append([i,scaleForData[dataNum_userStudy.get()-1][m],np.sqrt(error)/len(tmp_ySamples_sketch)])
#            #dissimilarity.append([i,overAllScale,np.sqrt(error)/len(tmp_ySamples_sketch)])
    
    if sketchRound==4:
        global extra_candidates
        extra_candidates=[]
        validIndex=[]
        while(1):
            tmpIndex=random.randint(0, len(dissimilarity)-1)
            if tmpIndex in validIndex:
                continue
            if dissimilarity[tmpIndex][6]-dissimilarity[tmpIndex][7]<=canvasHeight1-offsetH*2 and dissimilarity[tmpIndex][8]<=8:
                extra_candidates.append(dissimilarity[tmpIndex]+[-1])
                validIndex.append(tmpIndex)
            if len(validIndex)==10:
                break
    
    dissimilarity.sort(key=takeLast)
    global candidates
    candidates=[]
    #dissimilarity[0]
    for i in range(0,likert):
        if i!=likert-1:
            for j in range(0,int(candidatesNum/likert)):
                candidates.append(dissimilarity[int(i*1.0/(likert-1)*len(dissimilarity))+j*1]+[1-i*1.0/(likert-1)])
        else:
            for j in range(0,int(candidatesNum/likert)):
                candidates.append(dissimilarity[len(dissimilarity)-1-j*1]+[1-i*1.0/(likert-1)])
    
    global randomShown
    randomShown=[]
    
    while(1):
        tmp=random.randint(0, candidatesNum-1)
        if tmp not in randomShown:
            randomShown.append(tmp)
        if len(randomShown)==candidatesNum:
            break
    highlightMatchingResultFoUserStudy()
    
#    global minIndex
#    minIndex=dissimilarity.index(min(dissimilarity))
#    print(minIndex)
#    
#    global tmpDissimilarity
#    tmpDissimilarity=list(dissimilarity)
#    tmpDissimilarity.sort()

def query():

    global queryResults
    queryResults=[]
    left=originalPoints[:,1]
    
    left=left.reshape(-1,len(left),1)
    sketch=[]
    for i in range(0,len(ySamples_sketch)):
        sketch.append(ySamples_sketch[i])
    if len(sketch)<201:
        for i in range(0,201-len(sketch)):
            sketch.append(0)
    sketch=np.array(sketch)
    queryResults=model.predict([left,sketch.reshape(-1,len(sketch),1)])
    global resultMatrix
    resultMatrix = np.zeros((701, 2))
    #tmpMatrix = np.zeros((701,2))
    #matrix[0][1]=8
    for i in range(200,len(queryResults[0])):
        #tmpMatrix[i-200][0]=i-200+1
        #tmpMatrix[i-200][1]=queryResults[0][i]
        resultMatrix[i-200][0]=i-200+1
        resultMatrix[i-200][1]=queryResults[0][i]
    
    for i in range(0,len(resultMatrix)):
        for j in range(i+1,len(resultMatrix)):
            if resultMatrix[i][1]<resultMatrix[j][1]:
                tmpx=resultMatrix[i][0]
                tmpy=resultMatrix[i][1]
                resultMatrix[i][0]=resultMatrix[j][0]
                resultMatrix[i][1]=resultMatrix[j][1]
                resultMatrix[j][0]=tmpx
                resultMatrix[j][1]=tmpy
    
    btPredictionResult.config(state='normal')
    #sketch=np.array(trace).reshape(-1,len(trace),1)
    #for i in range(0,len(candidates)):
    #    print("compute:",i)
        #queryResults.append(np.array(candidates[i]).reshape(-1,len(candidates[i]),1))
    #    queryResults.append(model.predict([np.array(candidates[i]).reshape(-1,len(candidates[i]),1),sketch])[0][0])
    print("finished~~~~~~~~~~~~~~~~~~")
#    for position in range(0,len(allData)-1):
#        for step in range(position+1,len(allData)):
#            tmp=allData[position:step]
#            print(position)
#            print(step)
#            queryResults.append([position,step,model.predict([np.array(tmp).reshape(-1,len(tmp),1),sketch])[0][0]])
#
#    global maxIndex
#    for i in range(1,len(queryResults)):
#        if queryResults[i][2]>queryResults[maxIndex][2]:
#            maxIndex=i
#    print(queryResults[maxIndex][0])
#    print(queryResults[maxIndex][0]+queryResults[maxIndex][1])
#    print(interval)
#    for i in range(queryResults[maxIndex][0],queryResults[maxIndex][0]+queryResults[maxIndex][1]):
#        canvas1.create_line(i*interval, allData[i], (i+1)*interval, allData[i+1], fill='orange',tags='curve',joinstyle=tk.ROUND, width=1.5)      
def sketchPanelSwitch():
    if varSketch.get()==1:
        drawSketchPanel()
    else:
        canvas1.delete("curveInSketchPanel")
        deleteSketchPanel()
        clear()
def reset():
    deselectTheButton()
    global scaleAndShiftRecord
    scaleAndShiftRecord=[]
    #global cumScale
    #cumScale=1
    global overAllScale
    overAllScale=1
    
    global dissimilarity
    dissimilarity=[]
    smoothData((k0[dataNum-1]-a[dataNum-1]*math.log(1,1.004))/(length*np.std(y)))
        
    btSmooth.config(from_=-k0[dataNum-1], to=k0[dataNum-1])
    btSmooth.set(0)        
canvas1 = tk.Canvas(window, bg='white', height=canvasHeight1, width=canvasWidth1)
canvas1.pack()
#canvas2 = tk.Canvas(window, bg='white', height=canvasHeight2, width=canvasWidth2)
#canvas2.pack()

def selectTheResult(val):
    print(val)
    var4.set(str('%.6f' % resultMatrix[int(val)-1][1]))
    print(resultMatrix[int(val)-1][1])
    
frame1 = tk.Frame(window)
frame2 = tk.Frame(window)
frame3 = tk.Frame(window)
frame4 = tk.Frame(window)
frame5 = tk.Frame(window)
frame6 = tk.Frame(window)
ttk.Label(frame1, text="Data:").grid(column=1, row=1)
dataName = tk.StringVar()
dataChosen = ttk.Combobox(frame1, width=12, textvariable=dataName, state='readonly')
dataChosen.grid(column=2,row=1)
dataChosen['values'] = ("goldPrice_m","weekly-demand-for-a-plastic-cont","oil-and-mining","monthly-closings-of-the-dowjones", "annual-common-stock-price-us-187","daily-foreign-exchange-rates-31-","monthly-boston-armed-robberies-j","numbers-on-unemployment-benefits","coloured-fox-fur-production-nain","chemical-concentration-readings")
dataChosen.current(0)
dataChosen.bind("<<ComboboxSelected>>",loadData)

btClear = tk.Button(frame1, text ="Clear",state='disabled',command=clear)
btSave = tk.Button(frame1, text ="Save",state='disabled',command=save)
btSliders = tk.Checkbutton(frame1, text='sliders', state='disabled',variable=var1, onvalue=1, offvalue=0,command=sliders)
varSketch = tk.IntVar()
btSketchPanel = tk.Checkbutton(frame1, text='SketchPanel', variable=varSketch, onvalue=1, offvalue=0,command=sketchPanelSwitch)
btQuery = tk.Button(frame1, text ="Query",state='disabled',command=query)
btScale = tk.Scale(frame2, orient=tk.HORIZONTAL, from_=0, to=100, resolution=0.033,length=600,command=smoothData)
btScale_sketch = tk.Scale(frame1, orient=tk.HORIZONTAL, state='disabled', from_=0, to=4, resolution=0.02,length=200,command=smoothSketch)
btScale_sketch.set(defaultSketchSmooth)
btQetch = tk.Button(frame1, text ="Comp",state='disabled',command=candidatesForUserStudyComp)
btClear.grid(row = 1, column = 3)
#btSave.grid(row = 1, column = 4)
#btSliders.grid(row = 1, column = 5)
btSketchPanel.grid(row = 1, column = 6)
#btQuery.grid(row = 1, column = 7)
btScale_sketch.grid(row = 1, column = 8)
btQetch.grid(row=1, column = 9)
#ttk.Label(frame2, text="Data smoothing:").grid(column=1, row=1)
#btScale.grid(row = 1, column = 2)

var = tk.StringVar()
var2 = tk.StringVar()
var3 = tk.StringVar()
var4 = tk.StringVar()
var5 = tk.StringVar()
btSmooth = tk.Scale(frame4, orient=tk.HORIZONTAL, from_=0, to=10, resolution=0.05,length=600, command=smoothData2)
#bt0 = tk.Radiobutton(frame3, text ="A", indicatoron=0, variable=var2, value='A', command=highlightMatchingResult)
#bt025 = tk.Radiobutton(frame3, text ="B", indicatoron=0, variable=var2, value='B', command=highlightMatchingResult)
#bt050 = tk.Radiobutton(frame3, text ="C", indicatoron=0, variable=var2, value='C', command=highlightMatchingResult)
#bt075 = tk.Radiobutton(frame3, text ="D", indicatoron=0, variable=var2, value='D', command=highlightMatchingResult)
#bt1 = tk.Radiobutton(frame3, text ="E", indicatoron=0, variable=var2, value='E', command=highlightMatchingResult)
lbDistanceValue = tk.Label(frame3,textvariable=var, bg='red',font=('Arial', 12), width=5)
dataNum_userStudy=tk.IntVar()
#data1=ttk.Radiobutton(frame3, text="1",variable=dataNum_userStudy, value=1, command=loadDataForUserStudy)
#data1.grid(column=7,row=1)
#data2=ttk.Radiobutton(frame3, text="2",variable=dataNum_userStudy, value=2, command=loadDataForUserStudy)
#data2.grid(column=8,row=1)
#data3=ttk.Radiobutton(frame3, text="3",variable=dataNum_userStudy, value=3, command=loadDataForUserStudy)
#data3.grid(column=9,row=1)
#data4=ttk.Radiobutton(frame3, text="4",variable=dataNum_userStudy, value=4, command=loadDataForUserStudy)
#data4.grid(column=10,row=1)
#data5=ttk.Radiobutton(frame3, text="5",variable=dataNum_userStudy, value=5, command=loadDataForUserStudy)
#data5.grid(column=11,row=1)


#ttk.Label(frame4, text="<less smoothing>").grid(column=1, row=1)
#btSmooth.grid(row = 1, column = 2)
#ttk.Label(frame4, text="<more smoothing>").grid(column=3, row=1)
btReset = tk.Button(frame4, text ="reset",command=reset)
#btReset.grid(row = 1, column = 4)

#bt0.grid(row = 1, column = 1)
#bt025.grid(row = 1, column = 2)
#bt050.grid(row = 1, column = 3)
#bt075.grid(row = 1, column = 4)
#bt1.grid(row = 1, column = 5)
#lbDistanceValue.grid(row = 1, column = 6)
#btScale.set(50)
#btScale.

rate_bt0 = tk.Radiobutton(frame5, text ="no match", indicatoron=0, variable=var3, value='A', command=finishedRate)
rate_bt025 = tk.Radiobutton(frame5, text ="bad match", indicatoron=0, variable=var3, value='B', command=finishedRate)
rate_bt050 = tk.Radiobutton(frame5, text ="half bad half good", indicatoron=0, variable=var3, value='C', command=finishedRate)
rate_bt075 = tk.Radiobutton(frame5, text ="good match", indicatoron=0, variable=var3, value='D', command=finishedRate)
rate_bt1 = tk.Radiobutton(frame5, text ="excellent", indicatoron=0, variable=var3, value='E', command=finishedRate)
rate_btSave = tk.Button(frame5, text ="Save",state='disabled',command=save)
lbUserStudyCount = tk.Label(frame5,textvariable=var5, bg='red',font=('Arial', 12), width=5)
ttk.Label(frame5, text="Rate:   ").grid(column=1, row=1)
rate_bt0.grid(row = 1, column = 2)
rate_bt025.grid(row = 1, column = 3)
rate_bt050.grid(row = 1, column = 4)
rate_bt075.grid(row = 1, column = 5)
rate_bt1.grid(row = 1, column = 6)
#rate_btSave.grid(row = 1, column = 7)
lbUserStudyCount.grid(row = 1, column = 8)

btPredictionResult = tk.Scale(frame6, orient=tk.HORIZONTAL, state='disabled',from_=1, to=canvasWidth1-(rightSketchPanel-leftSketchPanel)+1, resolution=1,length=canvasWidth1-(rightSketchPanel-leftSketchPanel)+1,command=selectTheResult)
#btPredictionResult.grid(row = 1, column = 1)
lbSimValue = tk.Label(frame6,textvariable=var4, bg='red',font=('Arial', 12), width=8)
#lbSimValue.grid(row = 1, column =2)
frame1.pack()
frame2.pack()
frame3.pack()
frame4.pack()
frame5.pack()
frame6.pack()
def setButtonDisabled():
    btClear.config(state='disabled')
    btScale_sketch.config(state='disabled')
    #btSliders.config(state='disabled')
    btSave.config(state='disabled')
    btQuery.config(state='disabled')
def setButtionAbled():
    btClear.config(state='normal')
    btScale_sketch.config(state='normal')
    btQetch.config(state='normal')
    if dataLoaded==True:
        btQuery.config(state='normal')
    if var1.get()==1:
        btSave.config(state='normal')
    
#def motion(event):
#    global pX,pY
#    global trace, traceX, traceY
#    endX, endY = event.x, event.y
#    if endX>pX:
#        trace.append(endY)
#        traceX.append(endX)
#        traceY.append(endY)
#    #trace.append([endX,endY])
#    #print('{}, {}'.format(endX, endY))
#        #canvas2.create_line(pX, pY, endX, endY, fill='red',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=3)
#        pX, pY = endX, endY

initX=0
initY=0    
#def click(event):
#    clear()
#    global finished
#    finished=False
#    global pX,pY
#    global trace,traceX,traceY
#    pX, pY = event.x, event.y
#    global initX,initY
#    initX=pX
#    initY=pY
#    #trace.append([pX,pY])
#    trace.append(pY)
#    traceX.append(pX)
#    traceY.append(pY)
#def release(event):
#    global finished
#    global sketchDelete
#    if initX!=event.x or initY!=event.y:
#        finished=True
#        setButtionAbled()
#        sketchDelete=False
#        salientPointsComp_sketch()
#        if dataLoaded==True:
#            global widthQ
#            widthQ=traceX[len(traceX)-1]-traceX[0]
#            compCandidates()
def compCandidates():
    global candidates
    candidates=[]
    for i in range(0,canvasWidth1-widthQ+1):
        tmpCandidates=[]
        valid=True
        for j in range(i,i+widthQ):
            if originalPoints[j][0]>=0 and originalPoints[j][0]<=canvasWidth1 and originalPoints[j][1]>=0 and originalPoints[j][1]<=canvasHeight1:
                tmpCandidates.append(originalPoints[j][1])
            else: 
                valid=False
                break
            
        if valid==True:
            candidates.append(tmpCandidates)
                

def expMovingAverage(values,window):
    weights=np.exp(np.linspace(-1.,0.,window))
    weights/=weights.sum()
    tmp = np.convolve(values,weights)[:len(values)]
    tmp[:window]=tmp[window]
    return tmp

def drawSmoothedSketchData():
    #canvas2.delete("sketchSplineCurve")
    #canvas2.delete("extrema_sketch")
    #canvas2.delete("inflectionPoints_sketch")
    canvas1.delete("sketchSplineCurve")
    canvas1.delete("extrema_sketch")
    canvas1.delete("inflectionPoints_sketch")
    global heightQ
    #widthQ=traceX[len(traceX)-1]-traceX[0]
    heightQ=ySamples_sketch.max()-ySamples_sketch.min()
    
    global y2_sketch
    y2_sketch=[]
    for i in range(0,len(xSamples_sketch)):
        y2_sketch.append(cubicSpline_sketch.derivatives(xSamples_sketch[i])[2])
        
#    for i in range(1,len(xSamples_sketch)):
# #       canvas2.create_line(xSamples_sketch[i], ySamples_sketch[i], xSamples_sketch[i-1], ySamples_sketch[i-1], fill='black',tags='sketchSplineCurve',joinstyle=tk.ROUND, width=1.5) 
#        canvas1.create_line(xSamples_sketch[i], ySamples_sketch[i], xSamples_sketch[i-1], ySamples_sketch[i-1], fill='black',tags='sketchSplineCurve',joinstyle=tk.ROUND, width=3.5) 
#           
    global inflectionPoints_sketch,extrema_sketch
    inflectionPoints_sketch=[]
    extrema_sketch=[]

#                
    min_idxs = argrelmin(ySamples_sketch)
    #print(len(min_idxs[0]))
    for i in range(0,len(min_idxs[0])):
        #canvas2.create_oval(xSamples_sketch[min_idxs[0][i]]-radius,ySamples_sketch[min_idxs[0][i]]-radius,xSamples_sketch[min_idxs[0][i]]+radius,ySamples_sketch[min_idxs[0][i]]+radius,fill='yellow', width=1.2, tags='extrema_sketch')
        #canvas1.create_oval(xSamples_sketch[min_idxs[0][i]]-radius,ySamples_sketch[min_idxs[0][i]]-radius,xSamples_sketch[min_idxs[0][i]]+radius,ySamples_sketch[min_idxs[0][i]]+radius,fill='yellow', width=1.2, tags='extrema_sketch')
        extrema_sketch.append(min_idxs[0][i])
#    #print(min_idxs)
    max_idxs = argrelmax(ySamples_sketch)
    for i in range(0,len(max_idxs[0])):
        #canvas2.create_oval(xSamples_sketch[max_idxs[0][i]]-radius,ySamples_sketch[max_idxs[0][i]]-radius,xSamples_sketch[max_idxs[0][i]]+radius,ySamples_sketch[max_idxs[0][i]]+radius,fill='red', width=1.2, tags='extrema_sketch')
        #canvas1.create_oval(xSamples_sketch[max_idxs[0][i]]-radius,ySamples_sketch[max_idxs[0][i]]-radius,xSamples_sketch[max_idxs[0][i]]+radius,ySamples_sketch[max_idxs[0][i]]+radius,fill='red', width=1.2, tags='extrema_sketch')
        extrema_sketch.append(max_idxs[0][i])
#    print(len(max_idxs[0]))
#     
    for i in range (1,len(y2_sketch)-1):
        if y2_sketch[i-1]*y2_sketch[i+1]<0:
            y2_sketch[i]=0
            #canvas2.create_oval(xSamples_sketch[i]-radius,ySamples_sketch[i]-radius,xSamples_sketch[i]+radius,ySamples_sketch[i]+radius,fill='blue', width=1.2, tags='inflectionPoints_sketch')
            #canvas1.create_oval(xSamples_sketch[i]-radius,ySamples_sketch[i]-radius,xSamples_sketch[i]+radius,ySamples_sketch[i]+radius,fill='blue', width=1.2, tags='inflectionPoints_sketch')
            inflectionPoints_sketch.append(i)
    #print("inflectionPoints_sketch",len(inflectionPoints_sketch))



    global salientPoints_sketch
    salientPoints_sketch_tmp=[0]
    salientPoints_sketch_tmp.extend(extrema_sketch)
    salientPoints_sketch_tmp.extend(inflectionPoints_sketch)
    #salientPoints_sketch_tmp=extrema_sketch+inflectionPoints_sketch
    salientPoints_sketch_tmp.sort()
    salientPoints_sketch_tmp.append(len(xSamples_sketch)-1)
    salientPoints_sketch=[]
    salientPoints_sketch.append(salientPoints_sketch_tmp[0])
    for i in range(1,len(salientPoints_sketch_tmp)):
        if abs(ySamples_sketch[salientPoints_sketch_tmp[i]]-ySamples_sketch[salientPoints_sketch[len(salientPoints_sketch)-1]])>0.01*heightQ:
            salientPoints_sketch.append(salientPoints_sketch_tmp[i])
#    for i in range(0,len(salientPoints_sketch)):
#        canvas2.create_oval(xSamples_sketch[salientPoints_sketch[i]]-radius,ySamples_sketch[salientPoints_sketch[i]]-radius,xSamples_sketch[salientPoints_sketch[i]]+radius,ySamples_sketch[salientPoints_sketch[i]]+radius,fill='blue', width=1.2, tags='inflectionPoints_sketch')


def salientPointsComp_sketch():
    global x_sketch,y_sketch,xSamples_sketch,ySamples_sketch
    global cubicSpline_sketch
    
    x_sketch=np.array(traceX,dtype=float)
    y_sketch=np.array(traceY,dtype=float)
    #y_sketch=expMovingAverage(y_sketch,20)
    #cubicSpline_sketch= interpolate.splrep(x_sketch,y_sketch)
    cubicSpline_sketch= UnivariateSpline(x_sketch,y_sketch)
    #print(cubicSpline_sketch.get_residual())
    cubicSpline_sketch.set_smoothing_factor(len(x_sketch)*np.std(y_sketch)*defaultSketchSmooth)
    #cubicSpline_sketch.set_smoothing_factor(0)
    #print(cubicSpline_sketch.get_residual())
    xSamples_sketch=np.linspace(traceX[0],traceX[len(traceX)-1],traceX[len(traceX)-1]-traceX[0]+1)
    ySamples_sketch= cubicSpline_sketch(xSamples_sketch)
    #ySamples_sketch=interpolate.splev(xSamples_sketch,cubicSpline_sketch,0)
    drawSmoothedSketchData()
    
#canvas2.bind('<B1-Motion>', motion)
#canvas2.bind('<Button-1>',click)
#canvas2.bind('<ButtonRelease-1>',release)

withinSketchPanel=False
def click2(event):
    global sel1, sel2, initX, initY
    error=2
    initX=event.x
    initY=event.y
    if event.x<=initPos1+error and event.x>=initPos1-error:
        sel1=True
        print("1")
    if event.x<=initPos2+error and event.x>=initPos2-error:
        sel2=True
        print("2")

#def motion2(event):
#    global initPos1,initPos2
#    global originalPoints,initX,initY
#    shift=np.array([event.x-initX,event.y-initY])
#    originalPoints+=shift
#    #global originalData
#    #originalData+=shift
#    global scaleAndShiftRecord
#    scaleAndShiftRecord.append([shift[0], shift[1], 0])
#    initX=event.x
#    initY=event.y
#    redraw()
#    if sel1==True and var1.get()==1:
#        initPos1=event.x
#        canvas1.delete("slide1")
#        canvas1.create_line(initPos1, 0, initPos1, canvasHeight1, fill='red',tags='slide1',joinstyle=tk.ROUND, width=1.5) 
#    if sel2==True and var1.get()==1:
#        initPos2=event.x
#        canvas1.delete("slide2")
#        canvas1.create_line(initPos2, 0, initPos2, canvasHeight1, fill='green',tags='slide2',joinstyle=tk.ROUND, width=1.5) 

def motion2(event):
    global initX,initY
    shift=np.array([event.x-initX,event.y-initY])    

    initX=event.x
    initY=event.y
    canvas1.delete("highLight")
    global tmpOriginalPoints
    if var2.get()=='A':
        tmpOriginalPoints+=shift
        for i in range(1,tmpOriginalPoints.shape[0]):
            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
        for i in range(1,N_sketch+1):
            canvas1.create_line(tmpOriginalPoints[i+minIndexOfData][0], tmpOriginalPoints[i+minIndexOfData][1], tmpOriginalPoints[i-1+minIndexOfData][0], tmpOriginalPoints[i-1+minIndexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
            
    if var2.get()=='B':
        tmpOriginalPoints+=shift
        for i in range(1,tmpOriginalPoints.shape[0]):
            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
        for i in range(1,N_sketch+1):
            canvas1.create_line(tmpOriginalPoints[i+indexOfData][0], tmpOriginalPoints[i+indexOfData][1], tmpOriginalPoints[i-1+indexOfData][0], tmpOriginalPoints[i-1+indexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
         
    if var2.get()=='C':
        tmpOriginalPoints+=shift
        for i in range(1,tmpOriginalPoints.shape[0]):
            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
        for i in range(1,N_sketch+1):
            canvas1.create_line(tmpOriginalPoints[i+indexOfData][0], tmpOriginalPoints[i+indexOfData][1], tmpOriginalPoints[i-1+indexOfData][0], tmpOriginalPoints[i-1+indexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
        
    if var2.get()=='D':
        tmpOriginalPoints+=shift
        for i in range(1,tmpOriginalPoints.shape[0]):
            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
        for i in range(1,N_sketch+1):
            canvas1.create_line(tmpOriginalPoints[i+indexOfData][0], tmpOriginalPoints[i+indexOfData][1], tmpOriginalPoints[i-1+indexOfData][0], tmpOriginalPoints[i-1+indexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
        
    if var2.get()=='E':
        tmpOriginalPoints+=shift
        for i in range(1,tmpOriginalPoints.shape[0]):
            canvas1.create_line(tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], tmpOriginalPoints[i-1][0], tmpOriginalPoints[i-1][1], fill='red',tags='highLight',joinstyle=tk.ROUND, width=1.5) 
        for i in range(1,N_sketch+1):
            canvas1.create_line(tmpOriginalPoints[i+maxIndexOfData][0], tmpOriginalPoints[i+maxIndexOfData][1], tmpOriginalPoints[i-1+maxIndexOfData][0], tmpOriginalPoints[i-1+maxIndexOfData][1], fill='purple',tags='highLight',joinstyle=tk.ROUND, width=1.5) 

def release2(event):
    global sel1,sel2
    sel1=False
    sel2=False
    #print(finished)

def mouseMotion(event):
    global currentX, currentY
    currentX=event.x
    currentY=event.y
    
    global scaleAndShiftRecord, scaled#, cumScale
    #if scaled==True:
        #scaleAndShiftRecord.append([scalePoint[0],scalePoint[1],cumScale])
        #cumScale=1
        #scaled=False

def click3(event):
    global initX, initY
    initX=event.x
    initY=event.y
    global pX,pY
    pX, pY = event.x, event.y
    global withinSketchPanel
    global trace, traceX, traceY
    #print(event.x,"   ",event.y)
    if varSketch.get()==1 and event.x>=leftSketchPanel and event.x<=rightSketchPanel and event.y<=canvasHeight1-offsetH and event.y>=offsetH:
        withinSketchPanel=True
#        trace.append(pY)
#        traceX.append(pX)
#        traceY.append(pY)
        
        clear()
    else:
        withinSketchPanel=False

sketchFlag=0
def motion3(event):
    global sketchFlag
    if varSketch.get()==1 and withinSketchPanel==True and event.x>=leftSketchPanel and event.x<=rightSketchPanel and event.y<=canvasHeight1-offsetH and event.y>=offsetH:
        global pX,pY
        global trace, traceX, traceY
        endX, endY = event.x, event.y
        
        if endX>pX:
            if sketchFlag==0:
                trace.append(pY)
                traceX.append(pX)
                traceY.append(pY)
                sketchFlag=1
            trace.append(endY)
            traceX.append(endX)
            traceY.append(endY)
            canvas1.create_line(pX, pY, endX, endY, fill='purple',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)
            pX, pY = endX, endY

def release3(event):
    global finished
    global sketchDelete
    global sketchNum
    global sketchFlag
    if withinSketchPanel==True:
        if initX!=event.x or initY!=event.y:
            finished=True
            setButtionAbled()
            sketchDelete=False
            sketchFlag=0
            salientPointsComp_sketch()
            sketchInfoComp()
            sketchNum+=1
            btScale_sketch.set(defaultSketchSmooth)
            if dataLoaded==True:
                global widthQ
                widthQ=traceX[len(traceX)-1]-traceX[0]
                #candidatesForUserStudyComp()
                #compCandidates()

#canvas1.bind('<ButtonRelease-1>',release2) 
#canvas1.bind('<Button-1>',click2)
#canvas1.bind('<B1-Motion>', motion2)
#canvas1.bind('<Motion>',mouseMotion)
canvas1.bind('<Button-1>',click3)
canvas1.bind('<B1-Motion>', motion3)
canvas1.bind('<ButtonRelease-1>',release3) 


def drawSketchPanel():
    if dataLoaded==True:
        redraw()
    canvas1.create_line(leftSketchPanel, 0+offsetH, leftSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
    canvas1.create_line(leftSketchPanel, 0+offsetH, rightSketchPanel, 0+offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
    canvas1.create_line(rightSketchPanel, 0+offsetH, rightSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5)) 
    canvas1.create_line(leftSketchPanel, canvasHeight1-offsetH, rightSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
    
def deleteSketchPanel():
    canvas1.delete("sketchPanel")

def redraw():
    canvas1.delete("curve")
    canvas1.delete("extrema")
    canvas1.delete("inflectionPoints")
    
    #canvas1.delete("tmpcurve")
#    if sketchDelete==False and dataLoaded==True:
#        compCandidates()      
    
#    tmpSpline = UnivariateSpline(originalData[:,0],originalData[:,1])
#    tmpSpline.set_smoothing_factor(length*np.std(originalData[:,1])*currentSmoothing)
    
    #tmp_xSamples=np.linspace(originalData[0][0],originalData[length-1][0],canvasWidth1+1)
    #tmp_ySamples=tmpSpline(tmp_xSamples)
    #tmp_ySamples=tmpSpline(xSamples)
    #currentPoints=np.c_[tmp_xSamples,tmp_ySamples]
    #currentPoints=np.c_[xSamples,tmp_ySamples]
    #for i in range(1,currentPoints.shape[0]):
    #    canvas1.create_line(currentPoints[i][0], currentPoints[i][1], currentPoints[i-1][0], currentPoints[i-1][1], fill='orange',tags='tmpcurve',joinstyle=tk.ROUND, width=1.5) 
    
    for i in range(1,originalPoints.shape[0]):
        canvas1.create_line(originalPoints[i][0], originalPoints[i][1], originalPoints[i-1][0], originalPoints[i-1][1], fill='black',tags='curve',joinstyle=tk.ROUND, width=1.5) 
    
    for i in range(0,len(extrema)):
        canvas1.create_oval(originalPoints[extrema[i]][0]-radius,originalPoints[extrema[i]][1]-radius,originalPoints[extrema[i]][0]+radius,originalPoints[extrema[i]][1]+radius,fill='yellow', width=1.2, tags='extrema')
    
    for i in range(0,len(inflectionPoints)):
        canvas1.create_oval(originalPoints[inflectionPoints[i]][0]-radius,originalPoints[inflectionPoints[i]][1]-radius,originalPoints[inflectionPoints[i]][0]+radius,originalPoints[inflectionPoints[i]][1]+radius,fill='blue', width=1.2, tags='inflectionPoints')

    global curveInSketchPanel_x, curveInSketchPanel_y
    
    if dataLoaded==True and varSketch.get()==1:
        #print("dsfdsfsdfsdfsdfsdfsdfsdfsdfsdfsdf")
        curveInSketchPanel_x=[]
        curveInSketchPanel_y=[]
        for i in range(0,originalPoints.shape[0]):
            if originalPoints[i][0]>=leftSketchPanel and originalPoints[i][0]<=rightSketchPanel and originalPoints[i][1]<=canvasHeight1-offsetH and originalPoints[i][1]>=offsetH:
                curveInSketchPanel_x.append(originalPoints[i][0])
                curveInSketchPanel_y.append(originalPoints[i][1])
    
        canvas1.delete("curveInSketchPanel")
        for i in range(1,len(curveInSketchPanel_x)):
            canvas1.create_line(curveInSketchPanel_x[i], curveInSketchPanel_y[i], curveInSketchPanel_x[i-1], curveInSketchPanel_y[i-1], fill='red',tags='curveInSketchPanel',joinstyle=tk.ROUND, width=1.5) 

    countCriticalPoints()
    highlightMatchingResult()
    
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
    
def deselectTheButton():
    bt0.deselect()
    bt025.deselect()
    bt050.deselect()
    bt075.deselect()
    bt1.deselect()
    var.set("")
    #lbDistanceValue.clipboard_clear()
    
def scaleCurve(event):
    deselectTheButton()
    global scaled, overAllScale
    scaled=True
    wheelStep=event.delta/120
    print(wheelStep)
    global currentScale#, cumScale
    scaleStep=1.05
    if wheelStep>0:
        currentScale=scaleStep
    if wheelStep<0:
        currentScale=1.0/scaleStep
    global originalPoints, scalePoint
    global scaleAndShiftRecord
    global btSmooth
    global scaleCount, maxScale
    global originalData
    global currentSmoothing
    if (k0[dataNum-1]-a[dataNum-1]*math.log(overAllScale*currentScale,1.004))>0:
        #cumScale*=currentScale
        overAllScale*=currentScale
        #print("cumScale:",cumScale)
    #canvas1.scale('all', currentX, currentY, currentScale, currentScale)
    #canvas1.scale('all', currentX, currentY, 1+wheelStep*0.1, 1+wheelStep*0.1)
        
        scalePoint=np.array([currentX,currentY])
        #originalData=scalePoint * (1 - currentScale) + originalData * currentScale
    
        #redraw()
        if(len(scaleAndShiftRecord)>0):
            if scaleAndShiftRecord[len(scaleAndShiftRecord)-1][2]!=0 and scalePoint[0]==scaleAndShiftRecord[len(scaleAndShiftRecord)-1][0] and scalePoint[1]==scaleAndShiftRecord[len(scaleAndShiftRecord)-1][1]:
                scaleAndShiftRecord[len(scaleAndShiftRecord)-1][2]*=currentScale
            else:
                scaleAndShiftRecord.append([scalePoint[0],scalePoint[1],currentScale])
        else:
            scaleAndShiftRecord.append([scalePoint[0],scalePoint[1],currentScale])
        currentSmoothing=(k0[dataNum-1]-a[dataNum-1]*math.log(overAllScale,1.004))/(length*np.std(y))
        smoothData(currentSmoothing)
        
        btSmooth.config(from_=a[dataNum-1]*math.log(overAllScale,1.004)-k0[dataNum-1], to=k0[dataNum-1])
        btSmooth.set(0)
        #print("from:",a[dataNum-1]*math.log(overAllScale,1.004)-k0[dataNum-1])
        #print("to:",k0[dataNum-1])

    else:
        if scaleCount==0:
            maxScale=overAllScale
            scaleCount=1
        print("only scale without changing smoothing")
        overAllScale*=currentScale
        scalePoint=np.array([currentX,currentY])
        originalPoints=scalePoint * (1 - currentScale) + originalPoints * currentScale
        #originalData=scalePoint * (1 - currentScale) + originalData * currentScale
        redraw()
        scaleAndShiftRecord.append([scalePoint[0],scalePoint[1],currentScale])
    print("overAllScale=",overAllScale)
    
canvas1.bind('<MouseWheel>', scaleCurve)



selectedTraining=[]
sketchTraining=[]
simTraining=[]
df = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy.csv')

trainingSize=0
def loadUserStudyData():
    global selectedTraining,sketchTraining, simTraining
    global trainingSize
    trainingSize=df.shape[0]
    global interval
    interval=10 #original data is 1

    for rowId, row in df.iterrows():
        tmpSelected=[]
        tmpSelected.append(row['selected'])
        tmpSketch=[]
        tmpSketch.append(row['sketch'])
#        tmpSim=[]
#        tmpSim.append(row['sim'])
        
        tmp=tmpSelected[0].split(";")
        
        tmpselectedTraining=[]
        tmpsketchTraining=[]
#        tmpsimTraining=[]
        #for i in range(0,int(len(tmp)/2)):
        #    tmpselectedTraining.append([float(tmp[2*i]),float(tmp[2*i+1])])
        for i in range(0,len(tmp),interval):
            tmpselectedTraining.append(float(tmp[i]))
        #print(len(tmp))
        tmp=tmpSketch[0][1:len(tmpSketch[0])].split(";")
        #tmp=tmpSketch[0].split(";")
        for i in range(0,len(tmp),interval):
            tmpsketchTraining.append(float(tmp[i]))
            
        #for i in range(0,int(len(tmp)/2)):
        #    tmpsketchTraining.append([float(tmp[2*i]),float(tmp[2*i+1])])
#        tmp=tmpSim[0].split(";")
#        for i in range(0,len(tmp)):
#            tmpsimTraining.append(float(tmp[i]))
        selectedTraining.append(tmpselectedTraining)
        sketchTraining.append(tmpsketchTraining)
#        simTraining.append(tmpsimTraining)

loadUserStudyData()

X_train=[]
Y_train=[]
max_seq_length=0

def trainingDataPreprocess():
    global X_train,Y_train,df,max_seq_length
    for rowId, row in df.iterrows():
        df.set_value(rowId, "selected",selectedTraining[rowId])
        df.set_value(rowId, "sketch",sketchTraining[rowId])
    X_train = {'left': df.selected, 'right': df.sketch}
    #X_train=np.array(X_train)

    X_train['left']=np.array(selectedTraining)
    X_train['right']=np.array(sketchTraining)
    #X_train['left']=X_train['left'].as_matrix(columns=None).reshape(1,901)
#    max_seq_length = max(df.sketch.map(lambda x: len(x)).max(), df.selected.map(lambda x: len(x)).max())
#    for dataset, side in itertools.product([X_train], ['left', 'right']):
#        dataset[side] = pad_sequences(dataset[side], dtype='float',maxlen=max_seq_length)
#    Y_train=np.array(simTraining)
    Y_train =df['sim']
    Y_train = Y_train.values

#trainingDataPreprocess()

def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=-1, keepdims=True))
    #return K.sum(K.abs(left-right), axis=1, keepdims=True)
#    x=left
#    y=right
#    x = K.l2_normalize(x, axis=-1)
#    y = K.l2_normalize(y, axis=-1)
#    return -K.mean(x * y, axis=-1, keepdims=True)
def manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.sum(K.abs(left-right), axis=1, keepdims=True)

def trainNetwork():
    batch_size = trainingSize
    #batch_size = 1
    global hunits
    hunits=20
    n_epoch = 40000
    gradient_clipping_norm = 1.25
    adam = Adam(lr=1e-4)
    shared_model = LSTM(hunits)
    #LSTM1=LSTM(hunits)
    #LSTM2=LSTM(hunits)
    # The visible layer
    left_input = Input(shape=(len(selectedTraining[0]),1), dtype='float')
    right_input = Input(shape=(len(sketchTraining[0]),1), dtype='float')
    left_output = shared_model(left_input)
    right_output = shared_model(right_input)
    malstm_distance = Lambda(function=lambda x: exponent_neg_manhattan_distance(x[0], x[1]),output_shape=lambda x: (x[0][0], 1))([left_output, right_output])
    #malstm_distance = ManDist()([left_output, right_output])
    
    #merged_vector = keras.layers.concatenate([left_output, right_output], axis=-1)
    #predictions = Dense(1, activation='sigmoid')(merged_vector)
    #model = Model(inputs=[left_input, right_input], outputs=predictions)
    
    model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])
    # Adadelta optimizer, with gradient clipping by norm
    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
    optimizer =RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
    model.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy'])
    # Start trainings
    training_start_time = time()
    malstm_trained = model.fit([X_train['left'].reshape(-1, len(selectedTraining[0]), 1), X_train['right'].reshape(-1, len(sketchTraining[0]), 1)], Y_train, batch_size=batch_size, epochs=n_epoch)
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))
    model.save('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/SiameseLSTM.h5')

#trainNetwork()
#model = load_model('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/SiameseLSTM.h5',custom_objects={'exponent_neg_manhattan_distance': exponent_neg_manhattan_distance})
#model.summary()

#x_test_left=X_train['left'].reshape(-1, len(selectedTraining[0]), 1)
#x_test_right=X_train['right'].reshape(-1, len(sketchTraining[0]), 1)
results=[]

def matchingQetch():
    global segments
    segments=len(salientPoints_sketch)

    global dissimilarity
    dissimilarity=[]
    #print(heightQ)
    for i in range(0,len(salientPoints)-segments+1):
        widthC=originalPoints[salientPoints[i+segments-1]][0]-originalPoints[salientPoints[i]][0]
        heightC=originalPoints[salientPoints[i]:salientPoints[i+segments-1]+1,1].max()-originalPoints[salientPoints[i]:salientPoints[i+segments-1]+1,1].min()
        #print(widthC)
        #print(heightC)
        Gx=widthC/widthQ
        Gy=heightC/heightQ
        LDE=0
        SE=0
        for j in range(1,segments):
            Rx=(originalPoints[salientPoints[i+j]][0]-originalPoints[salientPoints[i+j-1]][0])/(Gx*(xSamples_sketch[salientPoints_sketch[j]]-xSamples_sketch[salientPoints_sketch[j-1]]))
            Ry=abs(originalPoints[salientPoints[i+j]][1]-originalPoints[salientPoints[i+j-1]][1])/(Gy*abs(ySamples_sketch[salientPoints_sketch[j]]-ySamples_sketch[salientPoints_sketch[j-1]]))
            #print("Rx:",(originalPoints[salientPoints[i+j]][0]-originalPoints[salientPoints[i+j-1]][0]))
            LDE=LDE+math.pow(math.log(Rx),2)+math.pow(math.log(Ry),2)
            
            N=(int)((originalPoints[salientPoints[i+j]][0]-originalPoints[salientPoints[i+j-1]][0])/timeInterval)
            #print("N=",N)
            tmp_originalPoints=np.c_[xSamples, ySamples]
            xSamples_tmp=np.linspace(tmp_originalPoints[salientPoints[i+j-1]][0],tmp_originalPoints[salientPoints[i+j]][0],N+1)
            ySamples_tmp=cubicSpline(xSamples_tmp)
            tmp_originalPoints= np.c_[xSamples_tmp, ySamples_tmp]
            #print(tmp_originalPoints)
            for m in range(0,len(scaleAndShiftRecord)):
                if scaleAndShiftRecord[m][2]==0:
                    shift=np.array([scaleAndShiftRecord[m][0],scaleAndShiftRecord[m][1]])
                    tmp_originalPoints+=shift
                else:
                    scalePoint_tmp=np.array([scaleAndShiftRecord[m][0],scaleAndShiftRecord[m][1]])
                    tmp_originalPoints=scalePoint_tmp * (1 - scaleAndShiftRecord[m][2]) + tmp_originalPoints * scaleAndShiftRecord[m][2]
            ySamples_tmp=tmp_originalPoints[:,1]
			  #ySamples_tmp=interpolate.splev(xSamples_tmp,cubicSpline,0)
            
            xSamples_tmp_sketch=np.linspace(xSamples_sketch[salientPoints_sketch[j-1]],xSamples_sketch[salientPoints_sketch[j]],N+1)
            ySamples_tmp_sketch=cubicSpline_sketch(xSamples_tmp_sketch)
			  #ySamples_tmp_sketch=interpolate.splev(xSamples_tmp_sketch,cubicSpline_sketch,0)
            #localMax=max(ySamples_tmp[0],ySamples_tmp[N-1])
            #localMaxSketch=max(ySamples_tmp_sketch[0],ySamples_tmp_sketch[N-1])
            #print("localMinSketch",localMin)
            for points in range(0,N):
                SE=SE+abs(Gy*Ry*(canvasHeight1-ySamples_tmp_sketch[points])-(canvasHeight1-ySamples_tmp[points]))
            SE=SE/(N*heightC)
        #print("LDE:",LDE)
        #print("SE:",SE)
        dissimilarity.append(LDE+SE)
    global minIndex
    minIndex=dissimilarity.index(min(dissimilarity))
    print(minIndex)
    
    global tmpDissimilarity
    tmpDissimilarity=list(dissimilarity)
    tmpDissimilarity.sort()
    #print(tmpDissimilarity)
    bt0.select()
    bt0.invoke()
   
X_test = {'left': df.selected, 'right': df.sketch}
prediction=[]

def predict():
    #model = load_model('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/SiameseLSTM.h5')
    model.summary()

    #prediction = model.predict([X_test['left'][0], X_test['right'][0]])
    testSize=len(X_test['left'])
    global prediction
    prediction=[]
    for i in range(0,testSize):
        prediction.append(model.predict([np.array(X_test['left'][i]).reshape(-1,len(X_test['left'][i]),1), np.array(X_test['right'][i]).reshape(-1,len(X_test['right'][i]),1)]))
    
    #prediction = model.predict([s1, s2])
    print(prediction)
  


   
    

#X_test['left'][0].reshape(-1, len(X_test['left'][0]), 1)  
#trainNetwork()
#predict()
window.mainloop()