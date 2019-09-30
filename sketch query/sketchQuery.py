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
#from util import ManDist
class ManDist(Layer):
    """
    Keras Custom Layer that calculates Manhattan Distance.
    """

    # initialize the layer, No need to include inputs parameter!
    def __init__(self, **kwargs):
        self.result = None
        super(ManDist, self).__init__(**kwargs)

    # input_shape will automatic collect input shapes to build layer
    def build(self, input_shape):
        super(ManDist, self).build(input_shape)

    # This is where the layer's logic lives.
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)
window = tk.Tk()
window.title('Sketching query for time series data')
window.geometry('920x550+500+200')

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
curveInSketchPanel_x=[]
curveInSketchPanel_y=[]
originalData=[]
currentSmoothing=0
offsetH=100
sketchPanelColor='#8CF2C2'
leftSketchPanel=350
rightSketchPanel=550
resultMatrix=[]
hunits = 20
interval=5
tmpOriginalPoints=[]
newTailY=[]
queryLength=40
from matplotlib import pyplot
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
#    global interval
#    interval=canvasWidth1*1.0/(length-1)
#    print(interval)
 
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
    
    canvas1.delete("results")
    canvas1.delete("resultsCurve")
    
    canvas2.delete("sketch")
    canvas2.delete("sketchSplineCurve")
    canvas2.delete("inflectionPoints_sketch")
    canvas2.delete("extrema_sketch")
    
    canvas1.delete("highLight")
    global sketchDelete
    sketchDelete=True
    btScale_sketch.set(1)
    global finished
    finished=False
    setButtonDisabled()
    global trace,traceX,traceY
    trace=[]
    traceX=[]
    traceY=[]
    global candidates
    candidates=[]
    
#def save():
#    global selected
#    selected=[]
#    if initPos1<=initPos2:
#        minR=initPos1
#        maxR=initPos2
#    else:
#        minR=initPos2
#        maxR=initPos1
#    
#    for i in range(0,length):
#        if i*interval>=minR and i*interval<=maxR:
#            selected.append(canvasHeight1-(dataset.iloc[i,1]-minDataY)/(maxDataY-minDataY)*canvasHeight1)
#            #selected.append([i*interval,canvasHeight1-(dataset.iloc[i,1]-minDataY)/(maxDataY-minDataY)*canvasHeight1])


def finishedRate():
    rate_btSave.config(state='normal')

#def save():
#    rate_btSave.config(state='disabled')
#    #print(var3.get())
#    global selected
#    selected=[]
#    
#    N=(int)((originalPoints[salientPoints[minIndex+segments-1]][0]-originalPoints[salientPoints[minIndex]][0])/timeInterval)
#    
#    tmp_originalPoints=np.c_[xSamples, ySamples]
#    xSamples_tmp=np.linspace(tmp_originalPoints[salientPoints[minIndex]][0],tmp_originalPoints[salientPoints[minIndex+segments-1]][0],N)
#    ySamples_tmp=cubicSpline(xSamples_tmp)
#    tmp_originalPoints= np.c_[xSamples_tmp, ySamples_tmp]
#    #print(tmp_originalPoints)
#    for m in range(0,len(scaleAndShiftRecord)):
#        if scaleAndShiftRecord[m][2]==0:
#            shift=np.array([scaleAndShiftRecord[m][0],scaleAndShiftRecord[m][1]])
#            tmp_originalPoints+=shift
#        else:
#            scalePoint_tmp=np.array([scaleAndShiftRecord[m][0],scaleAndShiftRecord[m][1]])
#            tmp_originalPoints=scalePoint_tmp * (1 - scaleAndShiftRecord[m][2]) + tmp_originalPoints * scaleAndShiftRecord[m][2]
#    ySamples_tmp=tmp_originalPoints[:,1]
#    
#    global count
#    count+=1
#    global userStudyData
#    tmpSelected=""
#    tmpTrace=""
#    for i in range(0,len(ySamples_tmp)):
#        if i==len(ySamples_tmp)-1:
#            tmpSelected=tmpSelected+str(ySamples_tmp[i])
#            #tmpSelected=tmpSelected+str(selected[i][0])+";"+str(selected[i][1])
#        else:
#            tmpSelected=tmpSelected+str(ySamples_tmp[i])+";"
#            #tmpSelected=tmpSelected+str(selected[i][0])+";"+str(selected[i][1])+";"
#    for i in range(0,len(trace)):
#        if i==len(trace)-1:
#            tmpTrace=tmpTrace+str(trace[i])
#            #tmpTrace=tmpTrace+str(trace[i][0])+";"+str(trace[i][1])
#        else:
#            tmpTrace=tmpTrace+str(trace[i])+";"
#            #tmpTrace=tmpTrace+str(trace[i][0])+";"+str(trace[i][1])+";"
#    userStudyData=[]
#    userStudyData.append(userId)
#    userStudyData.append(count)
#    userStudyData.append(originalPoints[salientPoints[minIndex]][0])
#    userStudyData.append(originalPoints[salientPoints[minIndex+segments-1]][0])
#    userStudyData.append(dataNum)
##    #userStudyData.append(selected)
##    #userStudyData.append(trace)
#    userStudyData.append(tmpSelected)
#    userStudyData.append(tmpTrace)
#    if var3.get()=='A':
#        userStudyData.append(0)
#    if var3.get()=='B':
#        userStudyData.append(0.25)
#    if var3.get()=='C':
#        userStudyData.append(0.5)
#    if var3.get()=='D':
#        userStudyData.append(0.75)
#    if var3.get()=='E':
#        userStudyData.append(1.0)
#
#    with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy.csv', 'a', newline='') as f:
#        writer = csv.writer(f)
#        writer.writerow(userStudyData)
#    print("saved")
tmp_xSamples=[]
tmp_ySamples=[]
def save():
    rate_btSave.config(state='disabled')
                
    N=rightSketchPanel-leftSketchPanel+1    
    
#    global curveInSketchPanel_y, curveInSketchPanel_x
#    curveInSketchPanel_y=np.array(curveInSketchPanel_y)
#    curveInSketchPanel_x=np.array(curveInSketchPanel_x)
    #tmpSpline= UnivariateSpline(curveInSketchPanel_x,curveInSketchPanel_y)
    tmpSpline= UnivariateSpline(originalPoints[:,0],originalPoints[:,1])
    global tmp_xSamples, tmp_ySamples
    tmp_xSamples=np.linspace(leftSketchPanel,rightSketchPanel,N)
    tmp_ySamples=tmpSpline(tmp_xSamples)
    
    
    global count
    count+=1
    global userStudyData
    tmpSelected=""
    tmpTrace=""
    tmpOriginalTraceX=""
    tmpOriginalTraceY=""

    for i in range(0,100):
        tmpSelected=tmpSelected+str(random.random()*canvasHeight1)+";"    
    for i in range(0,len(tmp_ySamples)):
        if i==len(tmp_ySamples)-1:
            tmpSelected=tmpSelected+str(tmp_ySamples[i])
        else:
            tmpSelected=tmpSelected+str(tmp_ySamples[i])+";"
    
    for i in range(0,len(ySamples_sketch)):
        if i==len(ySamples_sketch)-1:
            tmpTrace=tmpTrace+str(ySamples_sketch[i]) 
        else:
            tmpTrace=tmpTrace+str(ySamples_sketch[i])+";"
         
    if len(ySamples_sketch)<N:
        tmpTrace=tmpTrace+";"
        for i in range(0,201-len(ySamples_sketch)):
            if i==201-len(ySamples_sketch)-1:
                tmpTrace=tmpTrace+str("-1")
            else:
                tmpTrace=tmpTrace+str("-1;")
    
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

    with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(userStudyData)
    print("saved")  

def computeSimGroundTruth(val):
    simForTrain=""
    for i in range(1,202):
        simForTrain=simForTrain+str(val*i/201)+";"
    for i in range(200,0,-1):
        simForTrain=simForTrain+str(val*i/201)+";"
    for i in range(0,500):
        if i==500-1:
            simForTrain=simForTrain+str("0")
        else:
            simForTrain=simForTrain+str("0;")
    return simForTrain
        

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
    
def highlightMatchingResult():
    canvas1.delete("highLight")
    print(var2.get())
    global originalPoints
    if var2.get()=='A':
        minIndexOfData=dissimilarity.index(min(dissimilarity))
        print("dissimilarity:",min(dissimilarity))
        var.set(str('%.2f' % min(dissimilarity)))
        radiusH=radius+2
        
        tmpX=[]
        tmpY=[]
        for i in range(salientPoints[minIndexOfData],salientPoints[minIndexOfData+segments-1]+1):
            tmpX.append(originalPoints[i][0])
            tmpY.append(originalPoints[i][1])
        tmp_cubicSpline_data= UnivariateSpline(np.array(tmpX),np.array(tmpY))
        tmp_cubicSpline_data.set_smoothing_factor(0)
        
        xNewSamples=np.linspace(originalPoints[salientPoints[minIndexOfData]][0],originalPoints[salientPoints[minIndexOfData+segments-1]][0],int(originalPoints[salientPoints[minIndexOfData+segments-1]][0]-originalPoints[salientPoints[minIndexOfData]][0])+1)
        yNewSamples=tmp_cubicSpline_data(xNewSamples)
        
#        for i in range(0,segments):
#            tmpIndex=salientPoints[minIndexOfData+i]
#            canvas1.create_oval(originalPoints[tmpIndex][0]-radiusH,originalPoints[tmpIndex][1]-radiusH,originalPoints[tmpIndex][0]+radiusH,originalPoints[tmpIndex][1]+radiusH,fill='purple', width=2.2, tags='highLight')
        shift=np.array([(traceX[0]+traceX[len(traceX)-1])/2.0-(tmpX[0]+tmpX[len(tmpX)-1])/2.0,np.mean(traceY)-np.mean(yNewSamples)])
        
        qetchResult=np.c_[xNewSamples, yNewSamples]
        qetchResult+=shift
        
        for i in range(1,len(xNewSamples)):
            canvas1.create_line(qetchResult[i][0], qetchResult[i][1], qetchResult[i-1][0], qetchResult[i-1][1], fill='yellow',tags='highLight',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=3.5) 
        
        originalPoints+=shift
    
        global scaleAndShiftRecord
        scaleAndShiftRecord.append([shift[0], shift[1], 0])
        redraw()
    
#        canvas1.delete("results")
#        for i in range(localIndex-queryLength+1,localIndex+1):
#            canvas1.create_line(tmpOriginalPoints[i+1][0], tmpOriginalPoints[i+1][1], tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], fill='green',tags='results',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=3.5) 
#            canvas3.create_line(tmpOriginalPoints[i+1][0], tmpOriginalPoints[i+1][1], tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], fill='green',tags='results',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=3.5) 
    
    if var2.get()=='B':
        #canvas1.delete("highLight")
        indexOfData=dissimilarity.index(tmpDissimilarity[int(len(tmpDissimilarity)*0.25)])
        var.set(str('%.2f' % tmpDissimilarity[int(len(tmpDissimilarity)*0.25)]))
        radiusH=radius+2
        for i in range(0,segments):
            tmpIndex=salientPoints[indexOfData+i]
            canvas1.create_oval(originalPoints[tmpIndex][0]-radiusH,originalPoints[tmpIndex][1]-radiusH,originalPoints[tmpIndex][0]+radiusH,originalPoints[tmpIndex][1]+radiusH,fill='purple', width=2.2, tags='highLight')
        
    if var2.get()=='C':
        #canvas1.delete("highLight")
        indexOfData=dissimilarity.index(tmpDissimilarity[int(len(tmpDissimilarity)*0.5)])
        var.set(str('%.2f' % tmpDissimilarity[int(len(tmpDissimilarity)*0.5)]))
        radiusH=radius+2
        for i in range(0,segments):
            tmpIndex=salientPoints[indexOfData+i]
            canvas1.create_oval(originalPoints[tmpIndex][0]-radiusH,originalPoints[tmpIndex][1]-radiusH,originalPoints[tmpIndex][0]+radiusH,originalPoints[tmpIndex][1]+radiusH,fill='purple', width=2.2, tags='highLight')
        
    if var2.get()=='D':
        #canvas1.delete("highLight")
        indexOfData=dissimilarity.index(tmpDissimilarity[int(len(tmpDissimilarity)*0.75)])
        var.set(str('%.2f' % tmpDissimilarity[int(len(tmpDissimilarity)*0.75)]))
        radiusH=radius+2
        for i in range(0,segments):
            tmpIndex=salientPoints[indexOfData+i]
            canvas1.create_oval(originalPoints[tmpIndex][0]-radiusH,originalPoints[tmpIndex][1]-radiusH,originalPoints[tmpIndex][0]+radiusH,originalPoints[tmpIndex][1]+radiusH,fill='purple', width=2.2, tags='highLight')
        
    if var2.get()=='E':
        #canvas1.delete("highLight")
        maxIndexOfData=dissimilarity.index(max(dissimilarity))
        var.set(str('%.2f' % max(dissimilarity)))
        radiusH=radius+2
        for i in range(0,segments):
            tmpIndex=salientPoints[maxIndexOfData+i]
            canvas1.create_oval(originalPoints[tmpIndex][0]-radiusH,originalPoints[tmpIndex][1]-radiusH,originalPoints[tmpIndex][0]+radiusH,originalPoints[tmpIndex][1]+radiusH,fill='purple', width=2.2, tags='highLight')
        
def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=-1, keepdims=True))
    #return K.sum(K.abs(left-right), axis=1, keepdims=True)
#    x=left
#    y=right
#    x = K.l2_normalize(x, axis=-1)
#    y = K.l2_normalize(y, axis=-1)
#    return -K.mean(x * y, axis=-1, keepdims=True)
tmp_ySamples_sketch=[]
intervalNew=0
def sketchInfoComp():
    tmp_x_sketch=np.array(traceX,dtype=float)
    tmp_y_sketch=np.array(traceY,dtype=float)
    defaultSketchSmooth=0
    #defaultSketchSmooth=0.1
#    global N_sketch
#    N_sketch=int((traceX[len(traceX)-1]-traceX[0]))
    tmp_cubicSpline_sketch= UnivariateSpline(tmp_x_sketch,tmp_y_sketch)
    tmp_cubicSpline_sketch.set_smoothing_factor(len(x_sketch)*np.std(y_sketch)*defaultSketchSmooth)
    
    global tmp_ySamples_sketch
    #tmp_xSamples_sketch=np.linspace(traceX[0],traceX[len(traceX)-1], N_sketch+1)
    tmp_xSamples_sketch=np.linspace(traceX[0],traceX[len(traceX)-1], queryLength+1)
    tmp_ySamples_sketch= cubicSpline_sketch(tmp_xSamples_sketch)
    global intervalNew
    intervalNew=(traceX[len(traceX)-1]-traceX[0])/40.0
    
#    tmpLeft=[]
#    tmpRight=[]
#    for i in range(0,traceX[0]-leftSketchPanel):
#        tmpLeft.append(-1)
#    for i in range(0,rightSketchPanel-traceX[len(traceX)-1]):
#        tmpRight.append(-1)
#    
#    tmp_ySamples_sketch=np.append(tmpLeft,tmp_ySamples_sketch)
#    tmp_ySamples_sketch=np.append(tmp_ySamples_sketch,tmpRight)
    
model = load_model('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/SiameseLSTM.h5',custom_objects={'exponent_neg_manhattan_distance': exponent_neg_manhattan_distance})
model.summary()
inp = Input(batch_shape= (None,None, 1))
rnn= LSTM(hunits, return_sequences=True, name="RNN")(inp)
states = Model(inputs=[inp],outputs=[rnn])
states.summary()
states.layers[1].set_weights(model.layers[2].get_weights())     
rns1=[]
rns2=[]
tmp_ySamples=[]
def takeLast(elem):
    return elem[1]

queryResultsBeforeSort=[]
xQueryResult=[]


tmpvar1 = tk.StringVar()
tmpvar2 = tk.StringVar()
canvas3=[]
canvas4=[]
queried=False
window2=[]
def closeTheWindow():
    window2.destroy()
    global queried
    queried=False

def query():
    global queried
    queried=True
#    global window2
#    window2 = tk.Tk()
#    window2.title('comparison window')
#    window2.geometry('1820x460+50+0')
#    
#    tmpFrame1 = tk.Frame(window2)
#    tmpFrame2 = tk.Frame(window2)
#    global canvas3,canvas4
#    canvas3 = tk.Canvas(window2, bg='white', height=canvasHeight1, width=canvasWidth1)
#    canvas3.grid(row=1,column=1)
#    
#    tmp_bt0 = tk.Radiobutton(tmpFrame1, text ="no match", indicatoron=0, variable=tmpvar1, value='A')
#    tmp_bt025 = tk.Radiobutton(tmpFrame1, text ="bad match", indicatoron=0, variable=tmpvar1, value='B')
#    tmp_bt050 = tk.Radiobutton(tmpFrame1, text ="half bad half good", indicatoron=0, variable=tmpvar1, value='C')
#    tmp_bt075 = tk.Radiobutton(tmpFrame1, text ="good match", indicatoron=0, variable=tmpvar1, value='D')
#    tmp_bt1 = tk.Radiobutton(tmpFrame1, text ="excellent", indicatoron=0, variable=tmpvar1, value='E')
#    tmp_bt0.grid(row=1,column=1)
#    tmp_bt025.grid(row=1,column=2)
#    tmp_bt050.grid(row=1,column=3)
#    tmp_bt075.grid(row=1,column=4)
#    tmp_bt1.grid(row=1,column=5)
#    tmpFrame1.grid(row=2,column=1)
#    
#    
#    canvas4 = tk.Canvas(window2, bg='white', height=canvasHeight1, width=canvasWidth1)
#    canvas4.grid(row=1,column=2)
#    tmp2_bt0 = tk.Radiobutton(tmpFrame2, text ="no match", indicatoron=0, variable=tmpvar2, value='A')
#    tmp2_bt025 = tk.Radiobutton(tmpFrame2, text ="bad match", indicatoron=0, variable=tmpvar2, value='B')
#    tmp2_bt050 = tk.Radiobutton(tmpFrame2, text ="half bad half good", indicatoron=0, variable=tmpvar2, value='C')
#    tmp2_bt075 = tk.Radiobutton(tmpFrame2, text ="good match", indicatoron=0, variable=tmpvar2, value='D')
#    tmp2_bt1 = tk.Radiobutton(tmpFrame2, text ="excellent", indicatoron=0, variable=tmpvar2, value='E')
#    tmpLabel=tk.Label(tmpFrame2,width=8)
#    btFinished = tk.Button(tmpFrame2, text ="Finished",foreground="red",font='Helvetica 18 bold', command=closeTheWindow)
#    tmp2_bt0.grid(row=1,column=1)
#    tmp2_bt025.grid(row=1,column=2)
#    tmp2_bt050.grid(row=1,column=3)
#    tmp2_bt075.grid(row=1,column=4)
#    tmp2_bt1.grid(row=1,column=5)
#    tmpLabel.grid(row=1,column=6)
#    btFinished.grid(row=1,column=7)
#    tmpFrame2.grid(row=2,column=2)
#    
#    canvas3.create_line(leftSketchPanel, 0+offsetH, leftSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel2',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
#    canvas3.create_line(leftSketchPanel, 0+offsetH, rightSketchPanel, 0+offsetH, fill=sketchPanelColor,tags='sketchPanel2',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
#    canvas3.create_line(rightSketchPanel, 0+offsetH, rightSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel2',joinstyle=tk.ROUND, width=1.5,dash=(3,5)) 
#    canvas3.create_line(leftSketchPanel, canvasHeight1-offsetH, rightSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel2',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
#    
#    canvas4.create_line(leftSketchPanel, 0+offsetH, leftSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel3',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
#    canvas4.create_line(leftSketchPanel, 0+offsetH, rightSketchPanel, 0+offsetH, fill=sketchPanelColor,tags='sketchPanel3',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
#    canvas4.create_line(rightSketchPanel, 0+offsetH, rightSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel3',joinstyle=tk.ROUND, width=1.5,dash=(3,5)) 
#    canvas4.create_line(leftSketchPanel, canvasHeight1-offsetH, rightSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel3',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
#    
#    for i in range(1,len(traceX)):
#        canvas3.create_line(traceX[i], traceY[i], traceX[i-1], traceY[i-1], fill='#7F00FF',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=3)
#        canvas4.create_line(traceX[i], traceY[i], traceX[i-1], traceY[i-1], fill='#7F00FF',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=3)
    #btFinished.grid(row=3,column=1)
    #window2.destroy()
    
    
    
    global originalPoints
    global queryResults
    queryResults=[]
    global tmpOriginalPoints
    tmpOriginalPoints=originalPoints
    #widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
    #N_data=int(widthData)
    tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1]) 
    tmpmax=tmpOriginalPoints[0][0]
    count=0
    while(1):
        tmpmax+=intervalNew
        count+=1
        if tmpmax>tmpOriginalPoints[canvasWidth1][0]:
            break
    tmpmax-=intervalNew
    global tmp_xSamples,tmp_ySamples
    tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpmax,count)
    
    tmp_ySamples=tmpSpline(tmp_xSamples)
    
    left=[]
    #interval=1
    for i in range(1,len(tmp_ySamples)):
        left.append(tmp_ySamples[i]-tmp_ySamples[i-1])
    print("left:",len(left))
    tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
    
    left=np.array(left)
    left=left.reshape(-1,len(left),1)
    #print(left)
    
    sketchForPredict=[]
    for i in range(1,len(tmp_ySamples_sketch)):
        sketchForPredict.append(tmp_ySamples_sketch[i]-tmp_ySamples_sketch[i-1])
    sketchForPredict=np.array(sketchForPredict)
    sketch=sketchForPredict.reshape(-1,len(sketchForPredict),1)
    
    global rns1,rns2
    rns1 = states.predict(left)
    rns2 = states.predict(sketch)
    
    global queryResultsBeforeSort
    queryResultsBeforeSort=[]
#    for i in range(len(rns2[0])-1,len(rns1[0])):
#        queryResults.append([i-(len(rns2[0])-1),np.exp(-np.sum(np.abs(rns1[0][i,:]-rns2[0][len(rns2[0])-1,:])))])
    for i in range(0,len(rns1[0])):
        queryResultsBeforeSort.append(np.exp(-np.sum(np.abs(rns1[0][i,:]-rns2[0][len(rns2[0])-1,:]))))
        queryResults.append([i,np.exp(-np.sum(np.abs(rns1[0][i,:]-rns2[0][len(rns2[0])-1,:])))])
    
    queryResults=queryResults[queryLength-1:len(queryResults)]
    queryResults.sort(key=takeLast)
    
    maxSim=max(queryResultsBeforeSort)
    minSim=min(queryResultsBeforeSort)
    for i in range(0,len(queryResultsBeforeSort)):
        queryResultsBeforeSort[i]=(queryResultsBeforeSort[i]-minSim)/(maxSim-minSim)*canvasHeight1
    
    localIndex=queryResults[len(queryResults)-1][0]
    print("localIndex",localIndex)
    
    tmpY=[]
    for i in range(localIndex-queryLength+1,localIndex+2):
        tmpY.append(tmpOriginalPoints[i][1])
    
    shift=np.array([traceX[0]-tmpOriginalPoints[localIndex-queryLength+1][0],np.mean(traceY)-np.mean(tmpY)])
    originalPoints+=shift
    tmpOriginalPoints+=shift
    global scaleAndShiftRecord
    scaleAndShiftRecord.append([shift[0], shift[1], 0])
    redraw()
    
    canvas1.delete("results")
    for i in range(localIndex-queryLength+1,localIndex+1):
        canvas1.create_line(tmpOriginalPoints[i+1][0], tmpOriginalPoints[i+1][1], tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], fill='green',tags='results',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=5.5) 
        #canvas3.create_line(tmpOriginalPoints[i+1][0], tmpOriginalPoints[i+1][1], tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], fill='green',tags='results',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=3.5) 
    
    print("ndata:",count)

    
#    canvas1.delete("resultsCurve")
#    for i in range(2,len(tmp_xSamples)):
#        canvas1.create_line(tmpOriginalPoints[i][0], queryResultsBeforeSort[i-1], tmpOriginalPoints[i-1][0], queryResultsBeforeSort[i-2], fill='orange',tags='resultsCurve',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=3.5) 
      
    
    btPredictionResult.config(from_=1, to=len(queryResults))    
    btPredictionResult.config(state='normal')

    print("finished~~~~~~~~~~~~~~~~~~")

  
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
        
    btSmooth.config(from_=-k0[dataNum-1], to=4*k0[dataNum-1])
    btSmooth.set(0)        
canvas1 = tk.Canvas(window, bg='white', height=canvasHeight1, width=canvasWidth1)
canvas1.pack()
canvas2 = tk.Canvas(window, bg='white', height=canvasHeight2, width=canvasWidth2)
#canvas2.pack()

def selectTheResult(val):
    print(val)
    global originalPoints,tmpOriginalPoints
    #var4.set(str('%.6f' % resultMatrix[int(val)-1][1]))
    localIndex=queryResults[len(queryResults)-int(val)][0]
    
    tmpY=[]
    for i in range(localIndex-queryLength+1,localIndex+2):
        tmpY.append(tmpOriginalPoints[i][1])
    
    shift=np.array([traceX[0]-tmpOriginalPoints[localIndex-queryLength+1][0],np.mean(traceY)-np.mean(tmpY)])
    originalPoints+=shift
    tmpOriginalPoints+=shift
    global scaleAndShiftRecord
    scaleAndShiftRecord.append([shift[0], shift[1], 0])
    redraw()
    
    print("localIndex",localIndex)
    canvas1.delete("results")
    for i in range(localIndex-queryLength+1,localIndex+1):
        canvas1.create_line(tmpOriginalPoints[i+1][0], tmpOriginalPoints[i+1][1], tmpOriginalPoints[i][0], tmpOriginalPoints[i][1], fill='green',tags='results',joinstyle=tk.ROUND, width=3.5) 
   
    canvas1.delete("resultsCurve")
#    for i in range(2,len(tmp_xSamples)):
#        canvas1.create_line(tmpOriginalPoints[i][0], queryResultsBeforeSort[i-1], tmpOriginalPoints[i-1][0], queryResultsBeforeSort[i-2], fill='orange',tags='resultsCurve',joinstyle=tk.ROUND, width=3.5) 
    

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
btSketchPanel = tk.Checkbutton(frame1, text='SketchPanel', state='disabled',variable=varSketch, onvalue=1, offvalue=0,command=sketchPanelSwitch)
btQuery = tk.Button(frame1, text ="Query",state='disabled',command=query)
btScale = tk.Scale(frame2, orient=tk.HORIZONTAL, from_=0, to=100, resolution=0.033,length=600,command=smoothData)
btScale_sketch = tk.Scale(frame1, orient=tk.HORIZONTAL, state='disabled', from_=0, to=4, resolution=0.02,length=200,command=smoothSketch)
btScale_sketch.set(1)
btQetch = tk.Button(frame1, text ="Qetch",command=qetch)
btClear.grid(row = 1, column = 3)
#btSave.grid(row = 1, column = 4)
#btSliders.grid(row = 1, column = 5)
btSketchPanel.grid(row = 1, column = 6)
btQuery.grid(row = 1, column = 7)
#btScale_sketch.grid(row = 1, column = 8)
#btQetch.grid(row=1, column = 9)
#ttk.Label(frame2, text="Data smoothing:").grid(column=1, row=1)
#btScale.grid(row = 1, column = 2)

var = tk.StringVar()
var2 = tk.StringVar()
var3 = tk.StringVar()
var4 = tk.StringVar()
btSmooth = tk.Scale(frame4, orient=tk.HORIZONTAL, from_=0, to=10, resolution=0.05,length=600, command=smoothData2)
bt0 = tk.Radiobutton(frame3, text ="0%", indicatoron=0, variable=var2, value='A', command=highlightMatchingResult)
bt025 = tk.Radiobutton(frame3, text ="25%", indicatoron=0, variable=var2, value='B', command=highlightMatchingResult)
bt050 = tk.Radiobutton(frame3, text ="50%", indicatoron=0, variable=var2, value='C', command=highlightMatchingResult)
bt075 = tk.Radiobutton(frame3, text ="75%", indicatoron=0, variable=var2, value='D', command=highlightMatchingResult)
bt1 = tk.Radiobutton(frame3, text ="100%", indicatoron=0, variable=var2, value='E', command=highlightMatchingResult)
lbDistanceValue = tk.Label(frame3,textvariable=var, bg='red',font=('Arial', 12), width=5)

ttk.Label(frame4, text="<less smoothing>").grid(column=1, row=1)
btSmooth.grid(row = 1, column = 2)
ttk.Label(frame4, text="<more smoothing>").grid(column=3, row=1)
btReset = tk.Button(frame4, text ="reset",command=reset)
btReset.grid(row = 1, column = 4)
#bt0.grid(row = 1, column = 1)
#bt025.grid(row = 1, column = 2)
#bt050.grid(row = 1, column = 3)
#bt075.grid(row = 1, column = 4)
#bt1.grid(row = 1, column = 5)
#lbDistanceValue.grid(row = 1, column = 6)
#btScale.set(50)
#btScale.

rate_bt0 = tk.Radiobutton(frame5, text ="0%", indicatoron=0, variable=var3, value='A', command=finishedRate)
rate_bt025 = tk.Radiobutton(frame5, text ="25%", indicatoron=0, variable=var3, value='B', command=finishedRate)
rate_bt050 = tk.Radiobutton(frame5, text ="50%", indicatoron=0, variable=var3, value='C', command=finishedRate)
rate_bt075 = tk.Radiobutton(frame5, text ="75%", indicatoron=0, variable=var3, value='D', command=finishedRate)
rate_bt1 = tk.Radiobutton(frame5, text ="100%", indicatoron=0, variable=var3, value='E', command=finishedRate)
rate_btSave = tk.Button(frame5, text ="Save",state='disabled',command=save)
#ttk.Label(frame5, text="Rate:   ").grid(column=1, row=1)
#rate_bt0.grid(row = 1, column = 2)
#rate_bt025.grid(row = 1, column = 3)
#rate_bt050.grid(row = 1, column = 4)
#rate_bt075.grid(row = 1, column = 5)
#rate_bt1.grid(row = 1, column = 6)
#rate_btSave.grid(row = 1, column = 7)

btPredictionResult = tk.Scale(frame6, orient=tk.HORIZONTAL, state='disabled',from_=1, to=canvasWidth1-(rightSketchPanel-leftSketchPanel)+1, resolution=1,length=canvasWidth1-(rightSketchPanel-leftSketchPanel)+1,command=selectTheResult)
btPredictionResult.grid(row = 1, column = 1)
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
    if dataLoaded==True:
        btQuery.config(state='normal')
    if var1.get()==1:
        btSave.config(state='normal')
    
def motion(event):
    global pX,pY
    global trace, traceX, traceY
    endX, endY = event.x, event.y
    if endX>pX:
        trace.append(endY)
        traceX.append(endX)
        traceY.append(endY)
    #trace.append([endX,endY])
    #print('{}, {}'.format(endX, endY))
        canvas2.create_line(pX, pY, endX, endY, fill='red',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=3)
        pX, pY = endX, endY

initX=0
initY=0    
def click(event):
    clear()
    global finished
    finished=False
    global pX,pY
    global trace,traceX,traceY
    pX, pY = event.x, event.y
    global initX,initY
    initX=pX
    initY=pY
    #trace.append([pX,pY])
    trace.append(pY)
    traceX.append(pX)
    traceY.append(pY)
def release(event):
    global finished
    global sketchDelete
    if initX!=event.x or initY!=event.y:
        finished=True
        setButtionAbled()
        sketchDelete=False
        salientPointsComp_sketch()
        if dataLoaded==True:
            global widthQ
            widthQ=traceX[len(traceX)-1]-traceX[0]
            compCandidates()
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
    canvas2.delete("sketchSplineCurve")
    canvas2.delete("extrema_sketch")
    canvas2.delete("inflectionPoints_sketch")
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
#        canvas2.create_line(xSamples_sketch[i], ySamples_sketch[i], xSamples_sketch[i-1], ySamples_sketch[i-1], fill='black',tags='sketchSplineCurve',joinstyle=tk.ROUND, width=1.5) 
#        canvas1.create_line(xSamples_sketch[i], ySamples_sketch[i], xSamples_sketch[i-1], ySamples_sketch[i-1], fill='black',tags='sketchSplineCurve',joinstyle=tk.ROUND, width=1.5) 
           
    global inflectionPoints_sketch,extrema_sketch
    inflectionPoints_sketch=[]
    extrema_sketch=[]

#                
    min_idxs = argrelmin(ySamples_sketch)
    #print(len(min_idxs[0]))
    for i in range(0,len(min_idxs[0])):
#        canvas2.create_oval(xSamples_sketch[min_idxs[0][i]]-radius,ySamples_sketch[min_idxs[0][i]]-radius,xSamples_sketch[min_idxs[0][i]]+radius,ySamples_sketch[min_idxs[0][i]]+radius,fill='yellow', width=1.2, tags='extrema_sketch')
#        canvas1.create_oval(xSamples_sketch[min_idxs[0][i]]-radius,ySamples_sketch[min_idxs[0][i]]-radius,xSamples_sketch[min_idxs[0][i]]+radius,ySamples_sketch[min_idxs[0][i]]+radius,fill='yellow', width=1.2, tags='extrema_sketch')
        extrema_sketch.append(min_idxs[0][i])
#    #print(min_idxs)
    max_idxs = argrelmax(ySamples_sketch)
    for i in range(0,len(max_idxs[0])):
#        canvas2.create_oval(xSamples_sketch[max_idxs[0][i]]-radius,ySamples_sketch[max_idxs[0][i]]-radius,xSamples_sketch[max_idxs[0][i]]+radius,ySamples_sketch[max_idxs[0][i]]+radius,fill='red', width=1.2, tags='extrema_sketch')
#        canvas1.create_oval(xSamples_sketch[max_idxs[0][i]]-radius,ySamples_sketch[max_idxs[0][i]]-radius,xSamples_sketch[max_idxs[0][i]]+radius,ySamples_sketch[max_idxs[0][i]]+radius,fill='red', width=1.2, tags='extrema_sketch')
        extrema_sketch.append(max_idxs[0][i])
#    print(len(max_idxs[0]))
#     
    for i in range (1,len(y2_sketch)-1):
        if y2_sketch[i-1]*y2_sketch[i+1]<0:
            y2_sketch[i]=0
#            canvas2.create_oval(xSamples_sketch[i]-radius,ySamples_sketch[i]-radius,xSamples_sketch[i]+radius,ySamples_sketch[i]+radius,fill='blue', width=1.2, tags='inflectionPoints_sketch')
#            canvas1.create_oval(xSamples_sketch[i]-radius,ySamples_sketch[i]-radius,xSamples_sketch[i]+radius,ySamples_sketch[i]+radius,fill='blue', width=1.2, tags='inflectionPoints_sketch')
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


#    radiusH=radius+2
#    for i in range(0,1):
#        canvas1.create_oval(traceX[0]-radiusH,traceY[0]-radiusH,traceX[0]+radiusH,traceY[0]+radiusH,fill='purple', width=2.2, tags='bla')

def salientPointsComp_sketch():
    global x_sketch,y_sketch,xSamples_sketch,ySamples_sketch
    global cubicSpline_sketch
    
    x_sketch=np.array(traceX,dtype=float)
    y_sketch=np.array(traceY,dtype=float)
    #y_sketch=expMovingAverage(y_sketch,20)
    #cubicSpline_sketch= interpolate.splrep(x_sketch,y_sketch)
    cubicSpline_sketch= UnivariateSpline(x_sketch,y_sketch)
    #print(cubicSpline_sketch.get_residual())
    cubicSpline_sketch.set_smoothing_factor(len(x_sketch)*np.std(y_sketch)*1.0)
    #print(cubicSpline_sketch.get_residual())
    xSamples_sketch=np.linspace(traceX[0],traceX[len(traceX)-1],traceX[len(traceX)-1]-traceX[0]+1)
    ySamples_sketch= cubicSpline_sketch(xSamples_sketch)
    #ySamples_sketch=interpolate.splev(xSamples_sketch,cubicSpline_sketch,0)
    drawSmoothedSketchData()
    
canvas2.bind('<B1-Motion>', motion)
canvas2.bind('<Button-1>',click)
canvas2.bind('<ButtonRelease-1>',release)

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

def motion2(event):
    global initPos1,initPos2
    global originalPoints,initX,initY
    shift=np.array([event.x-initX,event.y-initY])
    originalPoints+=shift
    #global originalData
    #originalData+=shift
    global scaleAndShiftRecord
    scaleAndShiftRecord.append([shift[0], shift[1], 0])
    initX=event.x
    initY=event.y
    redraw()
    if sel1==True and var1.get()==1:
        initPos1=event.x
        canvas1.delete("slide1")
        canvas1.create_line(initPos1, 0, initPos1, canvasHeight1, fill='red',tags='slide1',joinstyle=tk.ROUND, width=1.5) 
    if sel2==True and var1.get()==1:
        initPos2=event.x
        canvas1.delete("slide2")
        canvas1.create_line(initPos2, 0, initPos2, canvasHeight1, fill='green',tags='slide2',joinstyle=tk.ROUND, width=1.5) 

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
    canvas1.delete("curveInSketchPanel")
    for i in range(1,len(curveInSketchPanel_x)):
        canvas1.create_line(curveInSketchPanel_x[i], curveInSketchPanel_y[i], curveInSketchPanel_x[i-1], curveInSketchPanel_y[i-1], fill='white',tags='curveInSketchPanel',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
    
    
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
    global sketchFlag
    if withinSketchPanel==True:
        if initX!=event.x or initY!=event.y:
            finished=True
            setButtionAbled()           
            sketchDelete=False
            salientPointsComp_sketch()
            sketchInfoComp()
            btScale_sketch.set(1)
            sketchFlag=0
            if dataLoaded==True:
                global widthQ
                widthQ=traceX[len(traceX)-1]-traceX[0]
                
                #compCandidates()

canvas1.bind('<ButtonRelease-1>',release2) 
canvas1.bind('<Button-1>',click2)
canvas1.bind('<B1-Motion>', motion2)
canvas1.bind('<Motion>',mouseMotion)
canvas1.bind('<Button-3>',click3)
canvas1.bind('<B3-Motion>', motion3)
canvas1.bind('<ButtonRelease-3>',release3) 


def drawSketchPanel():
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
        canvas1.create_line(originalPoints[i][0], originalPoints[i][1], originalPoints[i-1][0], originalPoints[i-1][1], fill='red',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
    
#    if queried==True:
#        for i in range(1,originalPoints.shape[0]):
#            canvas3.create_line(originalPoints[i][0], originalPoints[i][1], originalPoints[i-1][0], originalPoints[i-1][1], fill='red',tags='curve',joinstyle=tk.ROUND, width=1.5) 
#        for i in range(1,originalPoints.shape[0]):
#            canvas4.create_line(originalPoints[i][0], originalPoints[i][1], originalPoints[i-1][0], originalPoints[i-1][1], fill='red',tags='curve',joinstyle=tk.ROUND, width=1.5) 
    
#    for i in range(0,len(extrema)):
#        canvas1.create_oval(originalPoints[extrema[i]][0]-radius,originalPoints[extrema[i]][1]-radius,originalPoints[extrema[i]][0]+radius,originalPoints[extrema[i]][1]+radius,fill='yellow', width=1.2, tags='extrema')
#    
#    for i in range(0,len(inflectionPoints)):
#        canvas1.create_oval(originalPoints[inflectionPoints[i]][0]-radius,originalPoints[inflectionPoints[i]][1]-radius,originalPoints[inflectionPoints[i]][0]+radius,originalPoints[inflectionPoints[i]][1]+radius,fill='blue', width=1.2, tags='inflectionPoints')

    global curveInSketchPanel_x, curveInSketchPanel_y
    
    if dataLoaded==True and varSketch.get()==1:
        #print("dsfdsfsdfsdfsdfsdfsdfsdfsdfsdfsdf")
        curveInSketchPanel_x=[]
        curveInSketchPanel_y=[]
        for i in range(0,originalPoints.shape[0]):
            if originalPoints[i][0]>=leftSketchPanel and originalPoints[i][0]<=rightSketchPanel and originalPoints[i][1]<=canvasHeight1-offsetH and originalPoints[i][1]>=offsetH:
                curveInSketchPanel_x.append(originalPoints[i][0])
                curveInSketchPanel_y.append(originalPoints[i][1])
    
#        canvas1.delete("curveInSketchPanel")
#        for i in range(1,len(curveInSketchPanel_x)):
#            canvas1.create_line(curveInSketchPanel_x[i], curveInSketchPanel_y[i], curveInSketchPanel_x[i-1], curveInSketchPanel_y[i-1], fill='white',tags='curveInSketchPanel',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 

    countCriticalPoints()
    #highlightMatchingResult()
    
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
        
        btSmooth.config(from_=a[dataNum-1]*math.log(overAllScale,1.004)-k0[dataNum-1], to=4*k0[dataNum-1])
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
    
canvas1.bind('<MouseWheel>', scaleCurve)



selectedTraining=[]
originalSelectedTraining=[]
sketchTraining=[]
simTraining=[]
df = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy.csv')
df2 = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudyForTraining.csv')
df3 = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudyForTraining - Copy.csv')
df4 = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/sketchSynthesis.csv')
trainingSize=0
synthesizedData=[]

def loadSynthesisData():
    global synthesizedData
    synthesizedData=[]
    for rowId, row in df2.iterrows():
        tmpSelected=[]
        tmpSelected.append(row['sythesisY'])
            
        #print(tmpSelected[0])
        tmp=tmpSelected[0][1:len(tmpSelected[0])-1].split(";")
        #xSamples_random=np.linspace(250,349,100)
        y=[]
        
        for i in range(0,len(tmp)):
            y.append(float(tmp[i]))
        synthesizedData.append(y)
    
#loadSynthesisData()

def loadNewTail():
    global newTailY
    newTailY=[]
    for rowId, row in df3.iterrows():
        tmpSelected=[]
        #tmpSelected.append(row['newTailY'])
        tmpSelected.append(row['diff'])
            
        #print(tmpSelected[0])
        tmp=tmpSelected[0][1:len(tmpSelected[0])-1].split(";")
        #xSamples_random=np.linspace(250,349,100)
        y=[]
        
        for i in range(0,len(tmp)):
            y.append(float(tmp[i]))
        newTailY.append(y)

#loadNewTail()
newSketchX=[]
newSketchY=[]
def loadSketches():
    global newSketchX,newSketchY
    newSketchX=[]
    newSketchY=[]
    for rowId, row in df4.iterrows():
        tmpX=[]
        tmpY=[]
        #tmpSelected.append(row['newTailY'])
        tmpX.append(row['newX'])
        tmpY.append(row['newY'])
        #print(tmpSelected[0])
        tmp=tmpX[0][1:len(tmpX[0])-1].split(";")
        #xSamples_random=np.linspace(250,349,100)
        x=[]
        y=[]
        
        for i in range(0,len(tmp)):
            x.append(float(tmp[i]))
        newSketchX.append(x)
        
        tmp=tmpY[0][1:len(tmpY[0])-1].split(";")
        for i in range(0,len(tmp)):
            y.append(float(tmp[i]))
        newSketchY.append(y)
        
loadSketches()
        
        
expanded=50
def loadUserStudyData():
    global selectedTraining,sketchTraining, simTraining#,tmp_xSamples_sketch
    global trainingSize
    #trainingSize=df.shape[0]
    #global interval
    #interval=1 #original data is 1
    global originalSelectedTraining
    index=0
    for rowId, row in df.iterrows():
        tmpSelected=[]
        tmpSelected.append(row['selected'])
        tmpSketch=[]
        tmpSketch.append(row['sketch'])
        tmpSketchX=[]
        tmpSketchX.append(row['sketch_x'])
        tmpSketchY=[]
        tmpSketchY.append(row['sketch_y'])
        
        tmp=tmpSketchX[0].split(";")
        tmpSketchXTraining=[]
        for i in range(0,len(tmp)):
            tmpSketchXTraining.append(float(tmp[i]))
        tmp=tmpSketchY[0].split(";")
        tmpSketchYTraining=[]
        for i in range(0,len(tmp)):
            tmpSketchYTraining.append(float(tmp[i]))
        tmpSketchXTraining=np.array(tmpSketchXTraining)
        tmpSketchYTraining=np.array(tmpSketchYTraining)
        
        #defaultSketchSmooth=0.1
#        defaultSketchSmooth=0
#        tmp_cubicSpline_sketch= UnivariateSpline(tmpSketchXTraining,tmpSketchYTraining)
#        tmp_cubicSpline_sketch.set_smoothing_factor(len(tmpSketchXTraining)*np.std(tmpSketchYTraining)*defaultSketchSmooth)
#        
#        tmp_xSamples_sketch=np.linspace(tmpSketchXTraining[0],tmpSketchXTraining[len(tmpSketchXTraining)-1], 41)
#        tmp_ySamples_sketch= tmp_cubicSpline_sketch(tmp_xSamples_sketch)
        
#        tmpsketchTraining=[]
#
#        for i in range(1,len(tmp_ySamples_sketch)):
#            tmpsketchTraining.append(tmp_ySamples_sketch[i]-tmp_ySamples_sketch[i-1])
            
        tmpsketchTrainingAll=[]
        for i in range(index*5,index*5+5):
            tmp_cubicSpline_sketch= UnivariateSpline(np.array(newSketchX[i]),np.array(newSketchY[i]))
            tmp_cubicSpline_sketch.set_smoothing_factor(0)
            tmp_xSamples_sketch=np.linspace(newSketchX[i][0],newSketchX[i][len(newSketchX[i])-1], 41)
            tmp_ySamples_sketch= tmp_cubicSpline_sketch(tmp_xSamples_sketch)
            
            tmpsketchTraining=[]

            for i in range(1,len(tmp_ySamples_sketch)):
                tmpsketchTraining.append(tmp_ySamples_sketch[i]-tmp_ySamples_sketch[i-1])
            tmpsketchTrainingAll.append(tmpsketchTraining)
        
        tmp=tmpSketch[0][1:len(tmpSketch[0])].split(";")
        tmpsketchTraining2=[]
        for i in range(0,len(tmp)):
            tmpsketchTraining2.append(float(tmp[i]))
            
        tmp1=0
        for i in range(0,len(tmpsketchTraining2)):
            if tmpsketchTraining2[i]==-1:
                tmp1+=1
            else:
                break
        startPoint=tmp1
        
        tmp2=0
        for i in range(0,len(tmpsketchTraining2)):
            if tmpsketchTraining2[i]!=-1:
                tmp2+=1
        sketchLength=tmp2
        
        tmp=tmpSelected[0].split(";")
        tmpselectedTrainingX=[]
        tmpselectedTrainingY=[]
        for i in range(100+startPoint,len(tmp)):
            tmpselectedTrainingY.append(float(tmp[i]))
            tmpselectedTrainingX.append(i+250)
            if len(tmpselectedTrainingY)==sketchLength:
                break
        tmp_cubicSpline_data=UnivariateSpline(np.array(tmpselectedTrainingX),np.array(tmpselectedTrainingY))
        tmp_cubicSpline_data.set_smoothing_factor(0)
        tmp_ySamples_data=tmp_cubicSpline_data(tmp_xSamples_sketch)
        tmpselectedTraining3=[]
        for i in range(1,len(tmp_ySamples_data)):
            tmpselectedTraining3.append(tmp_ySamples_data[i]-tmp_ySamples_data[i-1])
            
        
        
        for j in range(0,expanded):
            tmpselectedTraining=[]
            tmpselectedTraining2=[]
            for i in range(0,20):
                tmpselectedTraining.append(newTailY[index*expanded+j][i])
            
            tmpselectedTraining.extend(tmpselectedTraining3)
            
            for i in range(0,len(tmp),interval):
                tmpselectedTraining2.append(float(tmp[i]))
            selectedTraining.append(tmpselectedTraining)
            #sketchTraining.append(tmpsketchTraining)
            sketchTraining.append(tmpsketchTrainingAll[int(j/10)])
            originalSelectedTraining.append(tmpselectedTraining2)
        index+=1
    trainingSize=len(selectedTraining)
#        simTraining.append(tmpsimTraining)
        #trainingDataPreprocess()
            
#loadUserStudyData()

X_train=[]
Y_train=[]
max_seq_length=0

#def trainingDataPreprocess():
#    global X_train,Y_train,df,max_seq_length
#    for rowId, row in df.iterrows():
#        df.set_value(rowId, "selected",selectedTraining[rowId])
#        df.set_value(rowId, "sketch",sketchTraining[rowId])
#    X_train = {'left': df.selected, 'right': df.sketch}
#    max_seq_length = max(df.sketch.map(lambda x: len(x)).max(), df.selected.map(lambda x: len(x)).max())
#    for dataset, side in itertools.product([X_train], ['left', 'right']):
#        dataset[side] = pad_sequences(dataset[side], dtype='float',maxlen=max_seq_length)
#    Y_train =df['sim']
#    Y_train = Y_train.values

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
    newY=[]
    for i in range(0,len(Y_train)):
        for j in range(0,expanded):
            newY.append(Y_train[i])
    newY=np.array(newY)
    Y_train=newY

#trainingDataPreprocess()

#class ManDist(Layer):
#    """
#    Keras Custom Layer that calculates Manhattan Distance.
#    """
#    # initialize the layer, No need to include inputs parameter!
#    def __init__(self, **kwargs):
#        self.result = None
#        super(ManDist, self).__init__(**kwargs)
#    # input_shape will automatic collect input shapes to build layer
#    def build(self, input_shape):
#        super(ManDist, self).build(input_shape)
#    # This is where the layer's logic lives.
#    def call(self, x, **kwargs):
#        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
#        return self.result
#    # return output shape
#    def compute_output_shape(self, input_shape):
#        return K.int_shape(self.result)
def sigmoid(x):
    return(1.0/(1.0+np.exp(-x)))
def LSTMlayer(weight,x_t,h_tm1,c_tm1):
    '''
    c_tm1 = np.array([0,0]).reshape(1,2)
    h_tm1 = np.array([0,0]).reshape(1,2)
    x_t   = np.array([1]).reshape(1,1)
    
    warr.shape = (nfeature,hunits*4)
    uarr.shape = (hunits,hunits*4)
    barr.shape = (hunits*4,)
    '''
    warr,uarr, barr = weight
    s_t = (x_t.dot(warr) + h_tm1.dot(uarr) + barr)
    hunit = uarr.shape[0]
    i  = sigmoid(s_t[:,:hunit])
    f  = sigmoid(s_t[:,1*hunit:2*hunit])
    _c = np.tanh(s_t[:,2*hunit:3*hunit])
    o  = sigmoid(s_t[:,3*hunit:])
    c_t = i*_c + f*c_tm1
    h_t = o*np.tanh(c_t)
    return(h_t,c_t)

def manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.sum(K.abs(left-right), axis=1, keepdims=True)

def trainNetwork():
    batch_size = trainingSize
    #batch_size = 1
    global hunits
    hunits=20
    n_epoch = 250
    gradient_clipping_norm = 1.25
    adam = Adam(lr=1e-3)
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
    malstm_trained = model.fit([X_train['left'].reshape(-1, len(selectedTraining[0]), 1), X_train['right'].reshape(-1, len(sketchTraining[0]), 1)], Y_train, batch_size=batch_size, epochs=n_epoch,validation_split=0.2)
    training_end_time = time()
    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))
    model.save('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/SiameseLSTM.h5')
    pyplot.plot(malstm_trained.history['loss'])
    pyplot.plot(malstm_trained.history['val_loss'])
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.legend(['train', 'validation'], loc='upper right')
    pyplot.show()
#trainNetwork()


#x_test_left=X_train['left'].reshape(-1, len(selectedTraining[0]), 1)
#x_test_right=X_train['right'].reshape(-1, len(sketchTraining[0]), 1)
#results=[]

#Reproduce LSTM layer outputs by hands
#weightLSTM = model.layers[2].get_weights()
#warr,uarr, barr = weightLSTM
#warr.shape,uarr.shape,barr.shape
#for i in range(0,len(x_test_left)):
#    c_tm1 = np.array([0]*hunits).reshape(1,hunits)
#    h_tm1 = np.array([0]*hunits).reshape(1,hunits)
#    for j in range(0, len(x_test_left[i])):
#        x_t = x_test_left[i][j].reshape(1,1)
#        h_tm1,c_tm1 = LSTMlayer(weightLSTM,x_t,h_tm1,c_tm1)
#        
#    results1=h_tm1
#    
#    c_tm1 = np.array([0]*hunits).reshape(1,hunits)
#    h_tm1 = np.array([0]*hunits).reshape(1,hunits)
#    for j in range(0, len(x_test_right[i])):
#        x_t = x_test_right[i][j].reshape(1,1)
#        h_tm1,c_tm1 = LSTMlayer(weightLSTM,x_t,h_tm1,c_tm1)
#    
#    results2=h_tm1
#    results.append(np.exp(-np.sum(np.abs(results1-results2))))
#
#results=np.array(results)
#meanSquareError=np.sum(np.square(results-Y_train))/len(results)




#def trainSiameseNetwork():
#    batch_size = trainingSize
#    n_epoch = 50
#    n_hidden = 50
#    #gradient_clipping_norm = 1.25
#    
#    encoder_left =Sequential()
#    encoder_left.add(LSTM(n_hidden, input_shape=(max_seq_length, 1)))
#    encoder_right =Sequential()
#    encoder_right.add(LSTM(n_hidden, input_shape=(max_seq_length, 1)))
#    decoder=Sequential()
#    decoder = Sequential()
#    decoder.add(merge([encoder_left, encoder_right], mode='concat'))
#    decoder.add(Dense(32, activation='relu'))
#    decoder.add(Dense(1, activation='sigmoid'))
#    decoder.compile(loss='mean_squared_error',optimizer='rmsprop', metrics=['accuracy'])
#    decoder.fit([X_train['left'], X_train['right']], Y_train, batch_size=batch_size, epochs=n_epoch)  

 

#def matchingQetch():
#    timeInterval=1
#    global segments
#    segments=len(salientPoints_sketch)
#
#    global dissimilarity
#    dissimilarity=[]
#    #print(heightQ)
#    for i in range(0,len(salientPoints)-segments+1):
#        widthC=xSamples[salientPoints[i+segments-1]]-xSamples[salientPoints[i]]
#        heightC=ySamples[salientPoints[i]:salientPoints[i+segments-1]+1].max()-ySamples[salientPoints[i]:salientPoints[i+segments-1]+1].min()
#        Gx=widthC/widthQ
#        Gy=heightC/heightQ
#        LDE=0
#        SE=0
#        for j in range(1,segments):
#            Rx=(xSamples[salientPoints[i+j]]-xSamples[salientPoints[i+j-1]])/(Gx*(xSamples_sketch[salientPoints_sketch[j]]-xSamples_sketch[salientPoints_sketch[j-1]]))
#            Ry=abs(ySamples[salientPoints[i+j]]-ySamples[salientPoints[i+j-1]])/(Gy*abs(ySamples_sketch[salientPoints_sketch[j]]-ySamples_sketch[salientPoints_sketch[j-1]]))
#
#            LDE=LDE+math.pow(math.log(Rx),2)+math.pow(math.log(Ry),2)
#            
#            N=(int)((xSamples[salientPoints[i+j]]-xSamples[salientPoints[i+j-1]])/timeInterval)
#            xSamples_tmp=np.linspace(xSamples[salientPoints[i+j-1]],xSamples[salientPoints[i+j]],N)
#            ySamples_tmp=cubicSpline(xSamples_tmp)
#			  #ySamples_tmp=interpolate.splev(xSamples_tmp,cubicSpline,0)
#            
#            xSamples_tmp_sketch=np.linspace(xSamples_sketch[salientPoints_sketch[j-1]],xSamples_sketch[salientPoints_sketch[j]],N)
#            ySamples_tmp_sketch=cubicSpline_sketch(xSamples_tmp_sketch)
#			  #ySamples_tmp_sketch=interpolate.splev(xSamples_tmp_sketch,cubicSpline_sketch,0)
#            localMin=min(ySamples_tmp[0],ySamples_tmp[N-1])
#            localMinSketch=min(ySamples_tmp_sketch[0],ySamples_tmp_sketch[N-1])
#            for points in range(0,N):
#                SE=SE+abs(Gy*Ry*(ySamples_tmp_sketch[points]-localMinSketch)-(ySamples_tmp[points]-localMin))
#            SE=SE/(N*heightC)
#        dissimilarity.append(LDE+SE)
#    minIndex=dissimilarity.index(min(dissimilarity))
#    print(minIndex)
#    
#    global tmpDissimilarity
#    tmpDissimilarity=list(dissimilarity)
#    tmpDissimilarity.sort()
#    bt0.select()
#    bt0.invoke()

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
            
            for points in range(0,N+1):
                SE=SE+abs(Gy*Ry*(ySamples_tmp_sketch[points])+(ySamples_tmp[0]+ySamples_tmp[N])/2.0-(Gy*Ry*ySamples_tmp_sketch[0]+Gy*Ry*ySamples_tmp_sketch[N])/2.0-(ySamples_tmp[points]))
            SE=SE/(N*heightC)
        #print("LDE:",LDE)
        #print("SE:",SE)
        dissimilarity.append(LDE+SE)
    global minIndex
    minIndex=dissimilarity.index(min(dissimilarity))
    print("minindex:",minIndex)
    
    global tmpDissimilarity
    tmpDissimilarity=list(dissimilarity)
    tmpDissimilarity.sort()
    #print(tmpDissimilarity)
    bt0.select()
    bt0.invoke()
   

#def trainSiameseNetwork():
#    batch_size = trainingSize
#    n_epoch = 20
#    n_hidden = 200
#    #shared_model = create_base_network()
#    gradient_clipping_norm = 1.25
#
#    # The visible layer
#    left_input = Input(shape=(None,1), dtype='float')
#    right_input = Input(shape=(None,1), dtype='float')  
#    shared_model = LSTM(n_hidden)
#    left_output = shared_model(left_input)
#    right_output = shared_model(right_input)
#    
#    merged_vector = keras.layers.concatenate([left_output, right_output], axis=-1)
# 
#    # And add a logistic regression on top
#    predictions = Dense(1, activation='sigmoid')(merged_vector)
# 
#    # We define a trainable model linking the
#    # tweet inputs to the predictions
#    model = Model(inputs=[left_input, right_input], outputs=predictions)
#    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
#    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
#    model.summary()
#    #shared_model.summary()
#    # Start trainings
#    training_start_time = time()
#    model_trained = model.fit([X_train['left'].reshape(-1, max_seq_length, 1), X_train['right'].reshape(-1, max_seq_length, 1)], Y_train, batch_size=batch_size, epochs=n_epoch)
#    training_end_time = time()
#    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))   
#    model.save('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/SiameseLSTM.h5')
#def trainNetwork():
#    batch_size = trainingSize
#    n_epoch = 100
#    n_hidden = 50
#    OUTPUT_SIZE=1
#    #shared_model = create_base_network()
#    gradient_clipping_norm = 1.25
#
#    # The visible layer
#    left_input = Input(shape=(901,1), dtype='float')
#    right_input = Input(shape=(201,1), dtype='float')
#    left_model = LSTM(n_hidden, return_sequences=True)
#    right_model = LSTM(n_hidden)
#    #left_output =TimeDistributed(Dense(OUTPUT_SIZE))(left_model(left_input))
#    left_output =(left_model(left_input))
#    left_output = Flatten()(left_output)
#    #right_output = Dense(OUTPUT_SIZE)(right_model(right_input))
#    right_output = (right_model(right_input))
#    merged_vector = keras.layers.concatenate([left_output, right_output], axis=-1)
# 
#    # And add a logistic regression on top
#    predictions = Dense(canvasWidth1+1, activation='sigmoid')(merged_vector)
# 
#    # We define a trainable model linking the
#    # tweet inputs to the predictions
#    model = Model(inputs=[left_input, right_input], outputs=predictions)
#    optimizer = Adadelta(clipnorm=gradient_clipping_norm)
#    model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])
#    model.summary()
#    
#    # Start trainings
#    training_start_time = time()
#    model_trained = model.fit([X_train['left'].reshape(-1, canvasWidth1+1, 1), X_train['right'].reshape(-1, rightSketchPanel-leftSketchPanel+1, 1)], Y_train, batch_size=batch_size, epochs=n_epoch)
#    training_end_time = time()
#    print("Training time finished.\n%d epochs in %12.2f" % (n_epoch, training_end_time - training_start_time))   
#    model.save('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/SiameseLSTM.h5')


window.mainloop()