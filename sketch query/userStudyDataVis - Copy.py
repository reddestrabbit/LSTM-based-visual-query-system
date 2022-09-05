# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:54:26 2019

@author: Chaoran Fan
"""
import pandas as pd
import tkinter as tk
#from tkinter import ttk
import numpy as np
from scipy.interpolate import UnivariateSpline
import csv
from time import time
selectedTraining=[]
sketchTraining=[]
dataScale=[]
dataSetNum=[]
simTraining=[]
originalPoints2=[]
df = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy - Copy.csv')
trainingSize=0
interval=1
Y_train =df['sim']
Y_train = Y_train.values
dataNum=0
tmp_xSamples=[]
tmp_ySamples=[]
realTailData=[]
userIdForUserStudy=10

def save2():
    global userStudyData            
    userStudyData=[]
    tmpSelected=""
    N=rightSketchPanel-leftSketchPanel+1
    for i in range(0,N):
        if i==N-1:
            tmpSelected=tmpSelected+str(originalPoints2[i][1])
        else:
            tmpSelected=tmpSelected+str(originalPoints2[i][1])+";"
    tmpOriginalTraceX=""
    tmpOriginalTraceY=""
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
    
    tmpTrace="*"
    for i in range(0,len(tmp_ySamples_sketch)):
        if i==len(tmp_ySamples_sketch)-1:
            tmpTrace=tmpTrace+str(tmp_ySamples_sketch[i])
        else:
            tmpTrace=tmpTrace+str(tmp_ySamples_sketch[i])+";"
    
    userStudyData.append(userIdForUserStudy)
    userStudyData.append(index3)
    userStudyData.append(index2)#sketchNum
    userStudyData.append(tmpSelected)
    userStudyData.append(tmpTrace)
    userStudyData.append(tmpOriginalTraceX)
    userStudyData.append(tmpOriginalTraceY)
    userStudyData.append(end_time-start_time)
    userStudyData.append(extremaNum[order[userIdForUserStudy-1][index3-1]])
    #userStudyData.append(tmpRealData)
    with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy - Copy2.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(userStudyData)
    
    print("saved")


userId=[]
count=[]
sketchNum=[]
extremaNum=[]
comparePoint=[]
startPoint=[]
sketchLength=[]
order=[[0,7,17,20,35,48,60,72,84,94],[1,8,18,21,36,49,61,73,85,95],[86,9,37,22,19,96,74,62,2,50],[75,10,63,97,38,51,23,3,87,30],[39,11,64,98,4,88,24,76,52,31],[25,89,5,32,65,99,40,77,12,53],[6,33,26,13,90,70,66,54,78,41],[67,79,34,42,45,91,14,71,27,55],[68,28,43,92,80,58,15,56,82,46],[16,93,83,69,57,59,47,81,44,29]]

def loadUserStudyData():
    global selectedTraining,sketchTraining, simTraining
    global trainingSize
    global dataSetNum,dataScale
    trainingSize=df.shape[0]
    dataSetNum=df['dataSet']
    dataSetNum=dataSetNum.values
    dataScale=df['scale']
    dataScale=dataScale.values
    global userId, count, sketchNum, comparePoint, extremaNum
    userId=df['userId']
    count=df['count']
    sketchNum=df['sketchNum']
    comparePoint=df['comparePoint']
    extremaNum=df['extremaNum']
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

selectedData=[]
sketchX=[]
sketchY=[]
def loadUserStudyData2():
    global selectedData,sketchX,sketchY
    df2 = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy - Copy2.csv')
    for rowId, row in df2.iterrows():
        tmpSelected=[]
        tmpSelected.append(row['selected'])
        tmp=tmpSelected[0].split(";")
        tmpselectedTraining=[]
        for i in range(0,len(tmp)):
            tmpselectedTraining.append(float(tmp[i]))
        selectedData.append(tmpselectedTraining)
        
        tmpSketchX=[]
        tmpSketchX.append(row['sketch_x'])
        tmp=tmpSketchX[0].split(";")
        tmp_SketchX=[]
        for i in range(0,len(tmp)):
            tmp_SketchX.append(float(tmp[i]))
        sketchX.append(tmp_SketchX)
        
        tmpSketchY=[]
        tmpSketchY.append(row['sketch_y'])
        tmp=tmpSketchY[0].split(";")
        tmp_SketchY=[]
        for i in range(0,len(tmp)):
            tmp_SketchY.append(float(tmp[i]))
        sketchY.append(tmp_SketchY)
loadUserStudyData2()


samplesIndex=[]
synthesizedDataX=[]
synthesizedDataY=[]
tmpSynthesizedDataX=[]
tmpSynthesizedDataY=[]
xSamples_random=[]

index2=0
index3=0
def visualizeTheData(index):
    canvas1.delete("curve")
    canvas1.delete("sketch")
    canvas1.delete("randomCurve")
    canvas1.delete("realDataTail")
    canvas1.delete("newTail")
    global index3,xSamples_random
    #xSamples_realLeftTail=(150,349,200)
    xSamples=np.linspace(350,550,201)
    xSamples_random=np.linspace(250,349,100)
    #b=selectedTraining[index]
    global originalPoints2
    originalPoints2=np.c_[xSamples,selectedTraining[index][100:301]]

    for i in range(1,originalPoints2.shape[0]):
        #if sketchTraining[index][i]==-1 or sketchTraining[index][i-1]==-1:
            canvas1.create_line(originalPoints2[i][0], originalPoints2[i][1], originalPoints2[i-1][0], originalPoints2[i-1][1], fill='orange',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
        #else:
        #    canvas1.create_line(originalPoints2[i][0], originalPoints2[i][1], originalPoints2[i-1][0], originalPoints2[i-1][1], fill='orange',tags='curve',joinstyle=tk.ROUND, width=1.5)
   
    index3+=1
    global index2
    index2=0
    var1.set(str(index3))
    var2.set(str(""))
def discardTheData():
    global index2
    index2-=1
    var2.set(str(index2))
    canvas1.delete("sketch")
def nextData():
    visualizeTheData(order[userIdForUserStudy-1][index3])
index=0
def nextUserStudyData():
    canvas1.delete("sketch")
    canvas1.delete("curve")
    global index
    xSamples=np.linspace(350,550,201)
    originalPoints2=np.c_[xSamples,selectedData[index]]

    for i in range(1,originalPoints2.shape[0]):
        #if sketchTraining[index][i]==-1 or sketchTraining[index][i-1]==-1:
            canvas1.create_line(originalPoints2[i][0], originalPoints2[i][1], originalPoints2[i-1][0], originalPoints2[i-1][1], fill='orange',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
    
    for i in range(1,len(sketchX[index])):
        canvas1.create_line(sketchX[index][i], sketchY[index][i], sketchX[index][i-1], sketchY[index][i-1], fill='#7F00FF',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)
        
    index+=1
    
window = tk.Tk()
window.title('my window')
window.geometry('920x450+500+0')
var1 = tk.StringVar()
var2 = tk.StringVar()
canvasWidth1=900
canvasHeight1=400
canvas1 = tk.Canvas(window, bg='white', height=canvasHeight1, width=canvasWidth1)
canvas1.pack()
frame1 = tk.Frame(window)
btNext = tk.Button(frame1, text ="Next",command=nextData)
lbNum = tk.Label(frame1,textvariable=var1, bg='red',font=('Arial', 12), width=5)
lbSim = tk.Label(frame1,textvariable=var2, bg='yellow',font=('Arial', 12), width=5)
btDiscard = tk.Button(frame1, text ="Discard",command=discardTheData)
btNextUserStudy = tk.Button(frame1, text ="NextUserStudyData",command=nextUserStudyData)
btNext.grid(row=1,column=1)
lbNum.grid(row=1,column=2)
lbSim.grid(row=1,column=3)
btDiscard.grid(row=1,column=4)
btNextUserStudy.grid(row=1,column=5)
frame1.pack()
expanded=40
visualizeTheData(order[userIdForUserStudy-1][0])
leftSketchPanel=350
rightSketchPanel=550
offsetH=100
sketchPanelColor='#8CF2C2'
canvas1.create_line(leftSketchPanel, 0+offsetH, leftSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
canvas1.create_line(leftSketchPanel, 0+offsetH, rightSketchPanel, 0+offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
canvas1.create_line(rightSketchPanel, 0+offsetH, rightSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5)) 
canvas1.create_line(leftSketchPanel, canvasHeight1-offsetH, rightSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
from scipy.signal import argrelmin, argrelmax
heightQ=0
y2_sketch=[]
inflectionPoints_sketch=[]
extrema_sketch=[]
radius=2
def drawSmoothedSketchData():
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
        

    for i in range(1,len(xSamples_sketch)):
        canvas1.create_line(xSamples_sketch[i], ySamples_sketch[i], xSamples_sketch[i-1], ySamples_sketch[i-1], fill='black',tags='sketchSplineCurve',joinstyle=tk.ROUND, width=3.5) 


    global inflectionPoints_sketch,extrema_sketch
    inflectionPoints_sketch=[]
    extrema_sketch=[]

#                
    min_idxs = argrelmin(ySamples_sketch)
    #print(len(min_idxs[0]))
    for i in range(0,len(min_idxs[0])):        
        canvas1.create_oval(xSamples_sketch[min_idxs[0][i]]-radius,ySamples_sketch[min_idxs[0][i]]-radius,xSamples_sketch[min_idxs[0][i]]+radius,ySamples_sketch[min_idxs[0][i]]+radius,fill='yellow', width=1.2, tags='extrema_sketch')
        extrema_sketch.append(min_idxs[0][i])
#    #print(min_idxs)
    max_idxs = argrelmax(ySamples_sketch)
    for i in range(0,len(max_idxs[0])):
        canvas1.create_oval(xSamples_sketch[max_idxs[0][i]]-radius,ySamples_sketch[max_idxs[0][i]]-radius,xSamples_sketch[max_idxs[0][i]]+radius,ySamples_sketch[max_idxs[0][i]]+radius,fill='red', width=1.2, tags='extrema_sketch')
        extrema_sketch.append(max_idxs[0][i])
#    print(len(max_idxs[0]))
#     
    for i in range (1,len(y2_sketch)-1):
        if y2_sketch[i-1]*y2_sketch[i+1]<0:
            y2_sketch[i]=0
            canvas1.create_oval(xSamples_sketch[i]-radius,ySamples_sketch[i]-radius,xSamples_sketch[i]+radius,ySamples_sketch[i]+radius,fill='blue', width=1.2, tags='inflectionPoints_sketch')
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

defaultSketchSmooth=0.05
salientPoints_sketch=[]
x_sketch=[]
y_sketch=[]
xSamples_sketch=[]
ySamples_sketch=[]
cubicSpline_sketch=[]
tmp_ySamples_sketch=[]
def salientPointsComp_sketch():
    global x_sketch,y_sketch,xSamples_sketch,ySamples_sketch
    global cubicSpline_sketch
    
    x_sketch=np.array(traceX,dtype=float)
    y_sketch=np.array(traceY,dtype=float)
    cubicSpline_sketch= UnivariateSpline(x_sketch,y_sketch)
    cubicSpline_sketch.set_smoothing_factor(len(x_sketch)*np.std(y_sketch)*defaultSketchSmooth)

    xSamples_sketch=np.linspace(traceX[0],traceX[len(traceX)-1],traceX[len(traceX)-1]-traceX[0]+1)
    ySamples_sketch= cubicSpline_sketch(xSamples_sketch)
    
    tmpLeft=[]
    tmpRight=[]
    for i in range(0,traceX[0]-leftSketchPanel):
        tmpLeft.append(-1)
    for i in range(0,rightSketchPanel-traceX[len(traceX)-1]):
        tmpRight.append(-1)
    global tmp_ySamples_sketch
    tmp_ySamples_sketch=np.append(tmpLeft,ySamples_sketch)
    tmp_ySamples_sketch=np.append(tmp_ySamples_sketch,tmpRight)
    #drawSmoothedSketchData()

initX=0
initY=0
pX=0
pY=0
withinSketchPanel=False
trace=[]
traceX=[]
traceY=[]
sketchDelete=True
finished=False
def clear():
    canvas1.delete("sketch")
    canvas1.delete("sketchSplineCurve")
    canvas1.delete("inflectionPoints_sketch")
    canvas1.delete("extrema_sketch")
    
    global sketchDelete
    sketchDelete=True
    global finished
    finished=False
    global trace,traceX,traceY
    trace=[]
    traceX=[]
    traceY=[]
start_time=0
end_time=0
def click3(event):
    global start_time
    start_time = time()
    global initX, initY
    initX=event.x
    initY=event.y
    global pX,pY
    pX, pY = event.x, event.y
    global withinSketchPanel
    global trace, traceX, traceY

    if event.x>=leftSketchPanel and event.x<=rightSketchPanel and event.y<=canvasHeight1-offsetH and event.y>=offsetH:
        withinSketchPanel=True
        clear()
    else:
        withinSketchPanel=False

sketchFlag=0

def motion3(event):
    global sketchFlag
    if withinSketchPanel==True and event.x>=leftSketchPanel and event.x<=rightSketchPanel and event.y<=canvasHeight1-offsetH and event.y>=offsetH:
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
            canvas1.create_line(pX, pY, endX, endY, fill='#7F00FF',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)
            pX, pY = endX, endY

def release3(event):
    global finished
    global sketchDelete
    global end_time
    end_time = time()
    global sketchFlag
    global index2
    if withinSketchPanel==True:
        if initX!=event.x or initY!=event.y:
            finished=True
            sketchDelete=False
            sketchFlag=0
            index2+=1
            
            var2.set(str(index2))
            salientPointsComp_sketch()
            if index2<=12:
                save2()
#            if index2==12 and index3<10:
#                visualizeTheData(order[userIdForUserStudy-1][index3])

canvas1.bind('<Button-1>',click3)
canvas1.bind('<B1-Motion>', motion3)
canvas1.bind('<ButtonRelease-1>',release3) 

window.mainloop()