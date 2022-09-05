# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:54:26 2019

@author: Chaoran Fan
"""
import pandas as pd
import tkinter as tk
#from tkinter import ttk
import numpy as np
from pandas import read_csv
import math
from scipy.interpolate import UnivariateSpline
import random
import csv

selectedTraining=[]
sketchTraining=[]
dataScale=[]
dataSetNum=[]
simTraining=[]
originalPoints=[]
originalPoints2=[]
k0=[64218,20166,108728,72811,9054,549923,35271,39221,67734,109528]
a=[155,107,1371,300,27,1365,250,96,50,1280]
df = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy.csv')
df2 = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudyForTraining.csv')
df3 = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudyForTraining - Copy.csv') #new tail
trainingSize=0
interval=1
Y_train =df['sim']
Y_train = Y_train.values
dataNum=0
tmp_xSamples=[]
tmp_ySamples=[]
realTailData=[]
newTailX=[]
newTailY=[]
def save(sythesisDataX, sythesisDataY, index):
    global userStudyData            
    userStudyData=[]
    #print(sythesisData)
    tmpSythesisX="*"
    for i in range(0,len(sythesisDataX)):
        tmpSythesisX+=str(sythesisDataX[i])+";"
    tmpSythesisY="*"
    for i in range(0,len(sythesisDataY)):
        tmpSythesisY+=str(sythesisDataY[i])+";"
#    tmpRealData="*"
#    for i in range(0,len(realData)-1):
#        tmpRealData+=str(realData[i])+";"
    
    userStudyData.append(tmpSythesisX)
    userStudyData.append(tmpSythesisY)
    userStudyData.append(userId[index])
    userStudyData.append(count[index])
    userStudyData.append(sketchNum[index])
    #userStudyData.append(tmpRealData)
    with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudyForTraining.csv', 'a', newline='') as f:
    #with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudyForTraining2.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(userStudyData)
    
    print("saved")

def save2(newTailX, newTailY, newTailDiff,realTail,index):
    global userStudyData            
    userStudyData=[]
    #print(sythesisData)
    tmpNewTailX="*"
    for i in range(0,len(newTailX)):
        tmpNewTailX+=str(newTailX[i])+";"
    tmpNewTailY="*"
    for i in range(0,len(newTailY)):
        tmpNewTailY+=str(newTailY[i])+";"
    tmpNewTailDiff="*"
    for i in range(0,len(newTailDiff)):
        tmpNewTailDiff+=str(newTailDiff[i])+";"

    tmpRealTail="*"
    for i in range(0,len(realTail)):
        tmpRealTail+=str(realTail[i])+";"
    
    userStudyData.append(tmpNewTailX)
    userStudyData.append(tmpNewTailY)
    userStudyData.append(userId[index])
    userStudyData.append(count[index])
    userStudyData.append(sketchNum[index])
    userStudyData.append(tmpNewTailDiff)
    userStudyData.append(tmpRealTail)
    #userStudyData.append(tmpRealData)
    with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudyForTraining - Copy.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(userStudyData)
    
    print("saved")

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
    dataset = read_csv("C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/realData/"+currentDataSetName+".csv")
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

userId=[]
count=[]
sketchNum=[]
comparePoint=[]
startPoint=[]
sketchLength=[]
def loadUserStudyData():
    global selectedTraining,sketchTraining, simTraining
    global trainingSize
    global dataSetNum,dataScale
    trainingSize=df.shape[0]
    dataSetNum=df['dataSet']
    dataSetNum=dataSetNum.values
    dataScale=df['scale']
    dataScale=dataScale.values
    global userId, count, sketchNum, comparePoint
    userId=df['userId']
    count=df['count']
    sketchNum=df['sketchNum']
    comparePoint=df['comparePoint']
    for rowId, row in df.iterrows():
        tmpSelected=[]
        tmpSelected.append(row['selected'])
        tmpSketch=[]
        tmpSketch.append(row['sketch'])
        
        #print(tmpSelected[0])
        tmp=tmpSelected[0].split(";")
        
        tmpselectedTraining=[]
        tmpsketchTraining=[]

        for i in range(0,len(tmp),interval):
            tmpselectedTraining.append(float(tmp[i]))
        #print(len(tmp))
        tmp=tmpSketch[0][1:len(tmpSketch[0])].split(";")
        for i in range(0,len(tmp),interval):
            tmpsketchTraining.append(float(tmp[i]))
            
        selectedTraining.append(tmpselectedTraining)
        sketchTraining.append(tmpsketchTraining)

    for i in range(0,len(sketchTraining)):
        tmp1=0
        for j in range(0,len(sketchTraining[0])):
            if sketchTraining[i][j]==-1:
                tmp1+=1
            else:
                break
        startPoint.append(tmp1)
        tmp2=0
        for j in range(0,len(sketchTraining[0])):
            if sketchTraining[i][j]!=-1:
                tmp2+=1
        sketchLength.append(tmp2)

loadUserStudyData()
index=0

            
        
        
    

samplesIndex=[]
synthesizedDataX=[]
synthesizedDataY=[]
tmpSynthesizedDataX=[]
tmpSynthesizedDataY=[]
xSamples_random=[]
index2=0
def visualizeTheData():
    canvas1.delete("curve")
    canvas1.delete("sketch")
    canvas1.delete("randomCurve")
    canvas1.delete("realDataTail")
    canvas1.delete("newTail")
    global index,xSamples_random
    #xSamples_realLeftTail=(150,349,200)
    xSamples=np.linspace(350,550,201)
    xSamples_random=np.linspace(250,349,100)
    #b=selectedTraining[index]
    global originalPoints2
    originalPoints2=np.c_[xSamples,selectedTraining[index][100:301]]
    randomPoints=np.c_[xSamples_random,selectedTraining[index][0:100]]
    
#    for i in range(1,randomPoints.shape[0]):
#        canvas1.create_line(randomPoints[i][0], randomPoints[i][1], randomPoints[i-1][0], randomPoints[i-1][1], fill='black',tags='randomCurve',joinstyle=tk.ROUND, width=1.5) 
    
    for i in range(1,originalPoints2.shape[0]):
        if sketchTraining[index][i]==-1 or sketchTraining[index][i-1]==-1:
            #print(originalPoints2[i][0])
            if originalPoints2[i][0]<500:
                canvas1.create_line(originalPoints2[i][0], originalPoints2[i][1], originalPoints2[i-1][0], originalPoints2[i-1][1], fill='red',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
        else:
            canvas1.create_line(originalPoints2[i][0], originalPoints2[i][1], originalPoints2[i-1][0], originalPoints2[i-1][1], fill='orange',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)
   
    
    sketchPoints=np.c_[xSamples,sketchTraining[index]]
    tmpX_sketch=[]
    tmpY_sketch=[]
    #tmpSketch=[]
    for i in range(0,len(xSamples)):
        if sketchTraining[index][i]!=-1:
            tmpX_sketch.append(xSamples[i])
            tmpY_sketch.append(sketchTraining[index][i])
    tmpSplineSketch=UnivariateSpline(np.array(tmpX_sketch),np.array(tmpY_sketch))
    tmpSplineSketch.set_smoothing_factor(len(tmpX_sketch)*np.std(tmpY_sketch)*0.5)
    newY_sketch=tmpSplineSketch(np.array(tmpX_sketch))
    
           
    for i in range(1,len(newY_sketch)):
        canvas1.create_line(tmpX_sketch[i], newY_sketch[i], tmpX_sketch[i-1], newY_sketch[i-1], fill='pink',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
            
    for i in range(1,sketchPoints.shape[0]):
        if sketchPoints[i][1]!=-1 and sketchPoints[i-1][1]!=-1:
            canvas1.create_line(sketchPoints[i][0], sketchPoints[i][1], sketchPoints[i-1][0], sketchPoints[i-1][1], fill='purple',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
 
    global dataNum
    dataNum=dataSetNum[index]
    loadDataForUserStudy()
    scaleForUserStudy(dataScale[index])
    
    global tmp_xSamples, tmp_ySamples
    tmpOriginalPoints=originalPoints
    widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
    N_data=int(widthData)
    tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
    tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
    tmp_ySamples=tmpSpline(tmp_xSamples)
    tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
#    global samplesIndex
#    samplesIndex=[]
#    while(1): #8 samples
#        tmp=random.randint(0,N_data+1-201)
#        if tmp not in samplesIndex:
#            samplesIndex.append(tmp)
#        if len(samplesIndex)==expanded:
#            break
#    global synthesizedData
#    synthesizedData=np.c_[tmp_xSamples[samplesIndex[0]:samplesIndex[0]+201],tmp_ySamples[samplesIndex[0]:samplesIndex[0]+201]]
    global index2
    index2=0
    global tmpSynthesizedDataX, tmpSynthesizedDataY
    tmpSynthesizedDataX=synthesizedDataX[index*expanded:index*expanded+expanded]
    tmpSynthesizedDataY=synthesizedDataY[index*expanded:index*expanded+expanded]
    index+=1
    var1.set(str(index))
    var2.set(str(Y_train[index-1]))

def visualizeTheSynthesizedData():
    canvas1.delete("randomCurve")
    global index2
    global synthesizedData
    if index2<len(samplesIndex):
        synthesizedData=np.c_[tmp_xSamples[samplesIndex[index2]:samplesIndex[index2]+201],tmp_ySamples[samplesIndex[index2]:samplesIndex[index2]+201]]
        shift=[150-synthesizedData[0][0],originalPoints2[0][1]-synthesizedData[len(synthesizedData)-1][1]]
        synthesizedData+=shift
        for i in range(1,synthesizedData.shape[0]):
            canvas1.create_line(synthesizedData[i][0], synthesizedData[i][1], synthesizedData[i-1][0], synthesizedData[i-1][1], fill='black',tags='randomCurve',joinstyle=tk.ROUND, width=1.5) 
        index2+=1

def loadSynthesisData():
    global synthesizedDataX, synthesizedDataY
    synthesizedDataX=[]
    synthesizedDataY=[]
    for rowId, row in df2.iterrows():
        tmpSelected=[]
        tmpSelected.append(row['sythesisX'])
            
        #print(tmpSelected[0])
        tmp=tmpSelected[0][1:len(tmpSelected[0])-1].split(";")
        #xSamples_random=np.linspace(250,349,100)
        y=[]
        
        for i in range(0,len(tmp)):
            y.append(float(tmp[i]))
        synthesizedDataX.append(y)
        
        tmpSelected=[]
        tmpSelected.append(row['sythesisY'])
            
        tmp=tmpSelected[0][1:len(tmpSelected[0])-1].split(";")
        y=[]
        for i in range(0,len(tmp)):
            y.append(float(tmp[i]))
        synthesizedDataY.append(y)
        
loadSynthesisData()

def loadNewTail():
    global newTailX, newTailY
    newTailX=[]
    newTailY=[]
    for rowId, row in df3.iterrows():
        tmpSelected=[]
        tmpSelected.append(row['newTailX'])
            
        
        tmp=tmpSelected[0][1:len(tmpSelected[0])-1].split(";")
        #xSamples_random=np.linspace(250,349,100)
        y=[]
        
        for i in range(0,len(tmp)):
            y.append(float(tmp[i]))
        newTailX.append(y)
        
        tmpSelected=[]
        tmpSelected.append(row['newTailY'])
            
        tmp=tmpSelected[0][1:len(tmpSelected[0])-1].split(";")
        y=[]
        for i in range(0,len(tmp)):
            y.append(float(tmp[i]))
        newTailY.append(y)

loadNewTail()

def visualizeTheSynthesizedData2():
    global index2
    if index2<expanded:
        canvas1.delete("randomCurve")
        for i in range(1,len(tmpSynthesizedDataX[index2])):
            canvas1.create_line(tmpSynthesizedDataX[index2][i], tmpSynthesizedDataY[index2][i], tmpSynthesizedDataX[index2][i-1], tmpSynthesizedDataY[index2][i-1], fill='black',tags='randomCurve',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)
        #canvas1.create_line(350, originalPoints2[0][1], 349, tmpSynthesizedData[index2][99], fill='black',tags='randomCurve',joinstyle=tk.ROUND, width=1.5)
        index2+=1
    
def visualizeTheRealTail():
#    global dataNum
#    dataNum=dataSetNum[index-1]
#    loadDataForUserStudy()
#    scaleForUserStudy(dataScale[index-1])
#    tmpOriginalPoints=originalPoints
#    widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
#    N_data=int(widthData)
#    tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
#    tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
#    tmp_ySamples=tmpSpline(tmp_xSamples)
#    tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
    global realTailData
    if comparePoint[index-1]-int(sketchLength[index-1]/2)+1>=0:
        realTailData=np.c_[tmp_xSamples[comparePoint[index-1]-int(sketchLength[index-1]/2)+1:comparePoint[index-1]+1],tmp_ySamples[comparePoint[index-1]-int(sketchLength[index-1]/2)+1:comparePoint[index-1]+1]]
        shift=[leftSketchPanel+startPoint[index-1]-tmp_xSamples[comparePoint[index-1]],selectedTraining[index-1][startPoint[index-1]+100]-tmp_ySamples[comparePoint[index-1]]]
        realTailData+=shift
        canvas1.delete("realDataTail")
        for i in range(1, len(realTailData)):
            canvas1.create_line(realTailData[i][0], realTailData[i][1], realTailData[i-1][0], realTailData[i-1][1], fill='red',tags='realDataTail',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)
    else:
        #realTailData=np.c_[tmp_xSamples[0:comparePoint[index-1]+1],tmp_ySamples[0:comparePoint[index-1]+1]]
#        realTailData=np.c_[tmp_xSamples[0:int(sketchLength[index-1]/2)],tmp_ySamples[0:int(sketchLength[index-1]/2)]]
#        shift=[leftSketchPanel+startPoint[index-1]-tmp_xSamples[comparePoint[index-1]],selectedTraining[index-1][startPoint[index-1]+100]-tmp_ySamples[comparePoint[index-1]]]
#        realTailData+=shift
#        canvas1.delete("realDataTail")
#        for i in range(1, len(realTailData)):
#            canvas1.create_line(realTailData[i][0], realTailData[i][1], realTailData[i-1][0], realTailData[i-1][1], fill='green',tags='realDataTail',joinstyle=tk.ROUND, width=1.5)

        #tmp_realTailData=np.c_[tmp_xSamples[0:int(sketchLength[index-1]/2)],tmp_ySamples[0:int(sketchLength[index-1]/2)]]
        tmp_realTailData=np.c_[tmp_xSamples[comparePoint[index-1]:comparePoint[index-1]+int(sketchLength[index-1]/2)],tmp_ySamples[comparePoint[index-1]:comparePoint[index-1]+int(sketchLength[index-1]/2)]]
        newX=[]
        newY=[]
        for i in range(0,int(sketchLength[index-1]/2)):
            newX.append((tmp_realTailData[i][0] - tmp_xSamples[comparePoint[index-1]])*math.cos(math.pi) - (tmp_ySamples[comparePoint[index-1]] - tmp_realTailData[i][1])*math.sin(math.pi) + tmp_xSamples[comparePoint[index-1]])
            newY.append(-(tmp_realTailData[i][0] - tmp_xSamples[comparePoint[index-1]])*math.sin(math.pi) - (tmp_ySamples[comparePoint[index-1]] - tmp_realTailData[i][1])*math.cos(math.pi) + tmp_xSamples[comparePoint[index-1]])
        realTailData=np.c_[np.array(newX[::-1]),np.array(newY[::-1])]
        shift=[leftSketchPanel+startPoint[index-1]-newX[0],selectedTraining[index-1][startPoint[index-1]+100]-newY[0]]
        realTailData+=shift
        canvas1.delete("realDataTail")
        for i in range(1, len(realTailData)):
            canvas1.create_line(realTailData[i][0], realTailData[i][1], realTailData[i-1][0], realTailData[i-1][1], fill='red',tags='realDataTail',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)

newTail=[]
w1=[0,0.016,0.104,0.275,0.5,0.725,0.896,0.984,1]
w2=[1,0.984,0.896,0.725,0.5,0.275,0.104,0.016,0]
def visualizeTheBlending():
    #print("f")
    global newTail
    newTail=[]
    rangeForTail=np.linspace(realTailData[0][0],realTailData[len(realTailData)-1][0],20)
    tmpSplineRealTail=UnivariateSpline(realTailData[:,0],realTailData[:,1])
    tmpSplineRealTail.set_smoothing_factor(0)
    tmpSplineSynthesizedData=UnivariateSpline(np.array(tmpSynthesizedDataX[index2-1]),np.array(tmpSynthesizedDataY[index2-1]))
    tmpSplineSynthesizedData.set_smoothing_factor(0)
    realTail=tmpSplineRealTail(rangeForTail)
    print(realTail[len(realTail)-1])
    synthesizedData=tmpSplineSynthesizedData(rangeForTail)
    print(len(synthesizedData))
    newTailY=[]
    for i in range(0,11):
        newTailY.append(synthesizedData[i])
    for i in range(0,9):
        print(i)
        newTailY.append(synthesizedData[i+11]*w2[i]+ realTail[i+11]*w1[i])
    newTailY=np.array(newTailY)
    newTail=np.c_[rangeForTail,newTailY]
    canvas1.delete("newTail")
    for i in range(1, len(newTail)):
        canvas1.create_line(newTail[i][0], newTail[i][1], newTail[i-1][0], newTail[i-1][1], fill='green',tags='newTail',capstyle=tk.ROUND,width=5.5)
    
def visualizeTheBlending2():
    
    canvas1.delete("newTail")
    global newTail
    newTail=[]
    rangeForTail=np.linspace(realTailData[0][0],realTailData[len(realTailData)-1][0],20)
    tmpSplineRealTail=UnivariateSpline(realTailData[:,0],realTailData[:,1])
    tmpSplineRealTail.set_smoothing_factor(0)
    realTail=tmpSplineRealTail(rangeForTail)
    
    for k in range(0,expanded):
        tmpSplineSynthesizedData=UnivariateSpline(np.array(tmpSynthesizedDataX[k]),np.array(tmpSynthesizedDataY[k]))
        tmpSplineSynthesizedData.set_smoothing_factor(0)
    
        print(realTail[len(realTail)-1])
        synthesizedData=tmpSplineSynthesizedData(rangeForTail)
        print(len(synthesizedData))
        newTailY=[]
        for i in range(0,11):
            newTailY.append(synthesizedData[i])
        for i in range(0,9):
            print(i)
            newTailY.append(synthesizedData[i+11]*w2[i]+ realTail[i+11]*w1[i])
        newTailY=np.array(newTailY)
        newTail=np.c_[rangeForTail,newTailY]
    
        for i in range(1, len(newTail)):
            canvas1.create_line(newTail[i][0], newTail[i][1], newTail[i-1][0], newTail[i-1][1], fill='red',tags='newTail',joinstyle=tk.ROUND, width=1.5)
def visualizeTheBlending3(): 
    canvas1.delete("newTail")
    print("expanded",expanded)
    print("index",index)
    for k in range(0,expanded):
        for i in range(1, 21):
            canvas1.create_line(newTailX[(index-1)*expanded+k][i], newTailY[(index-1)*expanded+k][i], newTailX[(index-1)*expanded+k][i-1], newTailY[(index-1)*expanded+k][i-1], fill='green',tags='newTail',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)


window = tk.Tk()
window.title('my window')
window.geometry('920x650+500+0')
var1 = tk.StringVar()
var2 = tk.StringVar()
canvasWidth1=900
canvasHeight1=400
canvas1 = tk.Canvas(window, bg='white', height=600, width=canvasWidth1)
canvas1.pack()
frame1 = tk.Frame(window)
btNext = tk.Button(frame1, text ="Next",command=visualizeTheData)
lbNum = tk.Label(frame1,textvariable=var1, bg='red',font=('Arial', 12), width=5)
lbSim = tk.Label(frame1,textvariable=var2, bg='yellow',font=('Arial', 12), width=5)
btNextSynthesizedData = tk.Button(frame1, text ="NextSynthesizedData",command=visualizeTheSynthesizedData2)
btRealTail=tk.Button(frame1, text ="RealTail",command=visualizeTheRealTail)
btBlending=tk.Button(frame1, text ="Blending",command=visualizeTheBlending3)
btBlending2=tk.Button(frame1, text ="Blending2",command=visualizeTheBlending)
btNext.grid(row=1,column=1)
lbNum.grid(row=1,column=2)
lbSim.grid(row=1,column=3)
btNextSynthesizedData.grid(row=1,column=4)
btRealTail.grid(row=1, column=5)
btBlending.grid(row=1, column=6)
btBlending2.grid(row=1, column=7)
frame1.pack()
expanded=50
#visualizeTheData()
leftSketchPanel=350
rightSketchPanel=550
offsetH=100
sketchPanelColor='#8CF2C2'
canvas1.create_line(leftSketchPanel, 0+offsetH, leftSketchPanel, 300, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
canvas1.create_line(leftSketchPanel, 0+offsetH, rightSketchPanel, 0+offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
canvas1.create_line(rightSketchPanel, 0+offsetH, rightSketchPanel, 300, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5)) 
canvas1.create_line(leftSketchPanel, 300, rightSketchPanel, 300, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))

def generateTrainingData():
    global dataNum
    global tmp_xSamples,tmp_ySamples
    for i in range(0,len(selectedTraining)):
        dataNum=dataSetNum[i]
        loadDataForUserStudy()
        scaleForUserStudy(dataScale[i])
        tmpOriginalPoints=originalPoints
        widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
        N_data=int(widthData)
        tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
        tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
        tmp_ySamples=tmpSpline(tmp_xSamples)
        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
        
        tmpRange=[]
        for t in range(sketchLength[i]):
            tmpRange.append(comparePoint[i]+t)
        
#        paddingLeft=0
#        for j in range(0, len(sketchTraining[i])):
#            if sketchTraining[i][j]==-1:
#                paddingLeft+=1
#            else:
#                break
#        start=comparePoint[i]-paddingLeft   
#        print("start:",start)
        #print(1)
        samplesIndex=[]
        while(1): #8 samples
            tmp=random.randint(0,N_data+1-int(sketchLength[i]/2))
            print(tmp)
            valid=True
#            if tmp in tmpRange:
#                continue
            for j in range(len(samplesIndex)):
                if abs(tmp-samplesIndex[j])<10:
                    #print(abs(tmp-samplesIndex[j]))
                    valid=False
                    break
#            if tmp not in samplesIndex:
#                samplesIndex.append(tmp)
            if valid==True:
                samplesIndex.append(tmp)
            if len(samplesIndex)==expanded:
                break
        print(samplesIndex)
        for k in range(0,len(samplesIndex)):
            synthesizedData=np.c_[tmp_xSamples[samplesIndex[k]:samplesIndex[k]+int(sketchLength[i]/2)],tmp_ySamples[samplesIndex[k]:samplesIndex[k]+int(sketchLength[i]/2)]]
#            print(len(synthesizedData))
#            print(synthesizedData[len(synthesizedData)-1][1])
#            print(synthesizedData[0][0])
#            print(originalPoints2[0][1])
            shift1=[leftSketchPanel+startPoint[i]-synthesizedData[len(synthesizedData)-1][0],selectedTraining[i][startPoint[i]+100]-synthesizedData[len(synthesizedData)-1][1]]
            synthesizedData+=shift1
#            if start-200<0:
#                realData=np.c_[tmp_xSamples[start-200:start+1],tmp_ySamples[start-200:start+1]]
#            else:
#                realData=np.c_[tmp_xSamples[start-200:start+1],tmp_ySamples[start-200:start+1]]
            #print(tmp_xSamples[-1:10])
            #print(realData)
            #print("realData:",len(realData))
#            shift2=[250-realData[0][0],selectedTraining[i][100]-realData[len(realData)-1][1]]
#            realData+=shift2
            #save(synthesizedData[:,1], realData[:,1], i)
            save(synthesizedData[:,0],synthesizedData[:,1], i)
          
#generateTrainingData()
def generateTrainingData2():
    global dataNum
    global tmp_xSamples,tmp_ySamples
    for i in range(0,len(selectedTraining)):
        dataNum=dataSetNum[i]
        loadDataForUserStudy()
        scaleForUserStudy(dataScale[i])
        tmpOriginalPoints=originalPoints
        widthData= (tmpOriginalPoints[canvasWidth1][0]-tmpOriginalPoints[0][0])
        N_data=int(widthData)
        tmpSpline= UnivariateSpline(tmpOriginalPoints[:,0],tmpOriginalPoints[:,1])     
        tmp_xSamples=np.linspace(tmpOriginalPoints[0][0],tmpOriginalPoints[canvasWidth1][0],N_data+1)
        tmp_ySamples=tmpSpline(tmp_xSamples)
        tmpOriginalPoints= np.c_[tmp_xSamples, tmp_ySamples]
        
        if comparePoint[i]-int(sketchLength[i]/2)+1>=0:
            realTailData=np.c_[tmp_xSamples[comparePoint[i]-int(sketchLength[i]/2)+1:comparePoint[i]+1],tmp_ySamples[comparePoint[i]-int(sketchLength[i]/2)+1:comparePoint[i]+1]]
            shift=[leftSketchPanel+startPoint[i]-tmp_xSamples[comparePoint[i]],selectedTraining[i][startPoint[i]+100]-tmp_ySamples[comparePoint[i]]]
            realTailData+=shift
        else:
            tmp_realTailData=np.c_[tmp_xSamples[comparePoint[i]:comparePoint[i]+int(sketchLength[i]/2)],tmp_ySamples[comparePoint[i]:comparePoint[i]+int(sketchLength[i]/2)]]
            newX=[]
            newY=[]
            for j in range(0,int(sketchLength[i]/2)):
                newX.append((tmp_realTailData[j][0] - tmp_xSamples[comparePoint[i]])*math.cos(math.pi) - (tmp_ySamples[comparePoint[i]] - tmp_realTailData[j][1])*math.sin(math.pi) + tmp_xSamples[comparePoint[i]])
                newY.append(-(tmp_realTailData[j][0] - tmp_xSamples[comparePoint[i]])*math.sin(math.pi) - (tmp_ySamples[comparePoint[i]] - tmp_realTailData[j][1])*math.cos(math.pi) + tmp_xSamples[comparePoint[i]])
            realTailData=np.c_[np.array(newX[::-1]),np.array(newY[::-1])]
            shift=[leftSketchPanel+startPoint[i]-newX[0],selectedTraining[i][startPoint[i]+100]-newY[0]]
            realTailData+=shift
        
        rangeForTail=np.linspace(realTailData[0][0],realTailData[len(realTailData)-1][0],21)
        tmpSplineRealTail=UnivariateSpline(realTailData[:,0],realTailData[:,1])
        tmpSplineRealTail.set_smoothing_factor(0)
        realTail=tmpSplineRealTail(rangeForTail)

        for k in range(0,expanded):
            tmpSplineSynthesizedData=UnivariateSpline(np.array(synthesizedDataX[i*expanded+k]),np.array(synthesizedDataY[i*expanded+k]))
            tmpSplineSynthesizedData.set_smoothing_factor(0)
    
            #print(realTail[len(realTail)-1])
            synthesizedData=tmpSplineSynthesizedData(rangeForTail)
            #print(len(synthesizedData))
            newTailY=[]
            newTailDiff=[]
            realTailDiff=[]
            for j in range(0,12):
                newTailY.append(synthesizedData[j])
            for j in range(0,9):
                newTailY.append(synthesizedData[j+12]*w2[j]+ realTail[j+12]*w1[j])
            newTailY=np.array(newTailY)
            for j in range(1,len(newTailY)):
                newTailDiff.append(newTailY[j]-newTailY[j-1])
            for j in range(1,len(realTail)):
                realTailDiff.append(realTail[j]-realTail[j-1])
            newTailDiff=np.array(newTailDiff)
            newTail=np.c_[rangeForTail,newTailY]

            save2(newTail[:,0],newTail[:,1],newTailDiff,realTailDiff,i)

#generateTrainingData2()


window.mainloop()