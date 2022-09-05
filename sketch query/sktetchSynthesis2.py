# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 13:54:26 2019

@author: Chaoran Fan
"""
import pandas as pd
import tkinter as tk
import numpy as np
from scipy.interpolate import UnivariateSpline
import csv
from time import time
import math
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from scipy.special import gamma
import random
from gekko import GEKKO
from sklearn.metrics import r2_score
df = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudyTable.csv')

l1=df['L1']
angle1=df['angle1']
diff=df['diff']
l2=df['L2']
angle2=df['angle2']
deltaX=df['deltaX']
angle3=df['angle3']
absDiff=df['absDiff']
diffAngle=df['diffAngle']
midDiff=df['midDiff']
#x=np.c_[l1,angle1]
#y=np.array(diff)

#angle1_a=[]
#angle1_b=[]
#diff_a=[]
#diff_b=[]
#for i in range(len(angle1)):
#    if angle1[i]<=0:
#        angle1_a.append(angle1[i])
#        diff_a.append(diff[i])
#    else:
#        angle1_b.append(angle1[i])
#        diff_b.append(diff[i])
#
#angle1_a=np.array(angle1_a)
#angle1_b=np.array(angle1_b)
#diff_a=np.array(diff_a)
#diff_b=np.array(diff_b)
#
#m=GEKKO()
#x=m.Param(value=angle1_a)
#a=m.FV(value=0.1)
#b=m.FV(value=0.1)
#c=m.FV(value=0.1)
#a.STATUS=1
#b.STATUS=1
#c.STATUS=1
#
#y=m.CV(value=diff_a)
#y.FSTATUS=1
#
#m.Equation(y==1/(a*x+b)+c)
#m.options.IMODE=2
#m.solve()
#print(a.value[0])
#print(b.value[0])
#print(c.value[0])
#
#print(r2_score(y,diff_a))
#numStatis=[[0]*23]
#
#for i in range(len(deltaX)):
#    numStatis[0][deltaX[i]-1]+=1/len(deltaX)*5000
deltaXNum=[]
tmpNum=[]
tmpMidDiff=[]
while(1):
    tmp=random.randint(0, len(deltaX)-1)
    if tmp not in tmpNum:
        tmpNum.append(tmp)
        deltaXNum.append(deltaX[tmp])
    if len(deltaXNum)==5000:
        break
offset=min(deltaXNum)-1
deltaXNum=deltaXNum-offset

tmpNum=[]
while(1):
    tmp=random.randint(0, len(midDiff)-1)
    if tmp not in tmpNum:
        tmpNum.append(tmp)
        tmpMidDiff.append(midDiff[tmp])
    if len(tmpMidDiff)==5000:
        break

index=0
sampleDeltaX=[]
xSamples=np.linspace(350,550,201)
cubicSpline_data=[]
shift=0
array=[100,210,320,430,540,650,760,870,980,1090,1200,1310,1420,1530,1640,1750,1860,1970,2080,2190]
for i in range(0,20):
    for j in range(1,10):
        array.append(array[i]+j)

tmpX_sketch=[]
tmpY_sketch=[]
tmpX_curve=[]
tmpY_curve=[]
newY_sketch=[]
def nextUserStudyData():
    global index
    canvas1.delete("sketch")
    canvas1.delete("curve")
    canvas1.delete("synthesizedSketch")
    
    global tmpX_sketch,tmpY_sketch,tmpX_curve,tmpY_curve,newY_sketch
    tmpX_sketch=[]
    tmpY_sketch=[]
    tmpX_curve=[]
    tmpY_curve=[]
    for i in range(0,len(xSamples)):
        if sketchY[index][i]!=-1:
            tmpX_sketch.append(xSamples[i])
            tmpY_sketch.append(sketchY[index][i])
            tmpX_curve.append(xSamples[i])
            tmpY_curve.append(selectedData[index][100+i])
    
    tmpSplineSketch=UnivariateSpline(np.array(tmpX_sketch),np.array(tmpY_sketch))
    tmpSplineSketch.set_smoothing_factor(len(tmpX_sketch)*np.std(tmpY_sketch)*0.5)
    newY_sketch=tmpSplineSketch(np.array(tmpX_sketch))
    #originalPoints=np.c_[xSamples,selectedData[index][100:301]] #curve
    
#    if index in array:
#        Shift=tmpY_curve[0]-tmpY_sketch[0]
#    else:
#        Shift=newY_sketch[0]-tmpY_sketch[0]
    
    global cubicSpline_data,shift
    if index in array:
#        for i in range(1,originalPoints.shape[0]):
#            if sketchY[index][i]!=-1 and sketchY[index][i-1]!=-1:
#                canvas1.create_line(originalPoints[i][0], originalPoints[i][1], originalPoints[i-1][0], originalPoints[i-1][1], fill='red',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
        cubicSpline_data= UnivariateSpline(np.array(tmpX_curve),np.array(tmpY_curve))
        cubicSpline_data.set_smoothing_factor(0)
        shift=np.mean(tmpY_curve)-np.mean(tmpY_sketch)
        for i in range(1,len(tmpX_curve)):
            canvas1.create_line(tmpX_curve[i], tmpY_curve[i], tmpX_curve[i-1], tmpY_curve[i-1], fill='red',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=5.5) 
    else:
        print("pink")
        cubicSpline_data= UnivariateSpline(np.array(tmpX_sketch),np.array(newY_sketch))
        cubicSpline_data.set_smoothing_factor(0)
        shift=np.mean(newY_sketch)-np.mean(tmpY_sketch)
        for i in range(1,len(newY_sketch)):
            canvas1.create_line(tmpX_sketch[i], newY_sketch[i], tmpX_sketch[i-1], newY_sketch[i-1], fill='pink',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=5.5) 
    
    
    
    for i in range(1,len(tmpX_sketch)):
        canvas1.create_line(tmpX_sketch[i], tmpY_sketch[i]+shift, tmpX_sketch[i-1], tmpY_sketch[i-1]+shift, fill='purple',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=5.5)
    #print("variance",np.var(sketchY[index]))

    
    if index in array:
        index+=1
    else:
        index+=20
    var1.set(str(index))

def nextUserStudyData1():
    global index
#    canvas1.delete("sketch")
#    canvas1.delete("curve")
#    canvas1.delete("synthesizedSketch")
    
    global tmpX_sketch,tmpY_sketch,tmpX_curve,tmpY_curve,newY_sketch
    tmpX_sketch=[]
    tmpY_sketch=[]
    tmpX_curve=[]
    tmpY_curve=[]
    for i in range(0,len(xSamples)):
        if sketchY[index][i]!=-1:
            tmpX_sketch.append(xSamples[i])
            tmpY_sketch.append(sketchY[index][i])
            tmpX_curve.append(xSamples[i])
            tmpY_curve.append(selectedData[index][100+i])
    
    tmpSplineSketch=UnivariateSpline(np.array(tmpX_sketch),np.array(tmpY_sketch))
    tmpSplineSketch.set_smoothing_factor(len(tmpX_sketch)*np.std(tmpY_sketch)*0.5)
    newY_sketch=tmpSplineSketch(np.array(tmpX_sketch))
    #originalPoints=np.c_[xSamples,selectedData[index][100:301]] #curve
    
#    if index in array:
#        Shift=tmpY_curve[0]-tmpY_sketch[0]
#    else:
#        Shift=newY_sketch[0]-tmpY_sketch[0]
    
    global cubicSpline_data,shift
    if index in array:
#        for i in range(1,originalPoints.shape[0]):
#            if sketchY[index][i]!=-1 and sketchY[index][i-1]!=-1:
#                canvas1.create_line(originalPoints[i][0], originalPoints[i][1], originalPoints[i-1][0], originalPoints[i-1][1], fill='red',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
        shift=np.mean(tmpY_curve)-np.mean(tmpY_sketch)
        cubicSpline_data= UnivariateSpline(np.array(tmpX_curve),np.array(tmpY_curve))
        cubicSpline_data.set_smoothing_factor(0)
#        for i in range(1,len(tmpX_curve)):
#            canvas1.create_line(tmpX_curve[i], tmpY_curve[i], tmpX_curve[i-1], tmpY_curve[i-1], fill='red',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=5.5) 
    else:
        #print("pink")
        cubicSpline_data= UnivariateSpline(np.array(tmpX_sketch),np.array(newY_sketch))
        cubicSpline_data.set_smoothing_factor(0)
        shift=np.mean(newY_sketch)-np.mean(tmpY_sketch)
#        for i in range(1,len(newY_sketch)):
#            canvas1.create_line(tmpX_sketch[i], newY_sketch[i], tmpX_sketch[i-1], newY_sketch[i-1], fill='pink',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=5.5) 
#    
#    
#    
#    for i in range(1,len(tmpX_sketch)):
#        canvas1.create_line(tmpX_sketch[i], tmpY_sketch[i]+Shift, tmpX_sketch[i-1], tmpY_sketch[i-1]+Shift, fill='purple',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND, width=5.5)
    #print("variance",np.var(sketchY[index]))

    
    if index in array:
        index+=1
    else:
        index+=20
    var1.set(str(index))


#interval=0
newX=[]
newY=[]
cubicSpline_sketch=[]
def nextSythesis():

    global cubicSpline_sketch
    cubicSpline_sketch=UnivariateSpline(np.array(tmpX_sketch),np.array(tmpY_sketch))
    cubicSpline_sketch.set_smoothing_factor(0)
    
    global sampleDeltaX
    sampleDeltaX=[]
    minX=tmpX_sketch[0]
    maxX=tmpX_sketch[len(tmpX_sketch)-1]
    tmpCount=minX
    while(1):
        readSamples2(1)
        if tmpCount+sample_2[0]<maxX:
            sampleDeltaX.append(sample_2[0])
            tmpCount+=sample_2[0]
        #if tmpCount+sample_2[0]==maxX:
        else:
            sampleDeltaX.append(maxX-tmpCount)
            #tmpCount+=sample_2[0]
            break
    global newX,newY 
    newX=[tmpX_sketch[0]]
    newY=[tmpY_sketch[0]+shift]
    #oldX=[sketchX[index-1][0]]
    oldY=[tmpY_sketch[0]+shift]
    for i in range(len(sampleDeltaX)):
        #newX.append((newX[len(newX)-1]+newX[len(newX)-1]+sampleDeltaX[i])/2.0)
        newX.append(newX[len(newX)-1]+sampleDeltaX[i])
        tmpGoal=float(cubicSpline_data(newX[len(newX)-1]))
        oldY.append(float(cubicSpline_sketch(newX[len(newX)-1]))+shift)
        #angle1=math.atan2(tmpGoal-newY[len(newY)-1], sampleDeltaX[i])
        #angle2=math.atan2(oldY[len(oldY)-1]-oldY[len(oldY)-2], sampleDeltaX[i])
        
        tmpx1=np.array([sampleDeltaX[i],tmpGoal-newY[len(newY)-1]])
        
        #tmpx2=np.array([sampleDeltaX[i],oldY[len(oldY)-1]-oldY[len(oldY)-2]])
        tmpx2=np.array([sampleDeltaX[i],oldY[len(oldY)-1]-newY[len(newY)-1]])
        #print((np.sqrt(tmpx1.dot(tmpx1))*np.sqrt(tmpx2.dot(tmpx2))))
        #tmpDiffAngle=abs(angle1-angle2)
        tmp=tmpx1.dot(tmpx2)/(np.sqrt(tmpx1.dot(tmpx1))*np.sqrt(tmpx2.dot(tmpx2)))
        if tmp>1:
            tmp=1
        if tmp<-1:
            tmp=-1
        tmpDiffAngle=np.arccos(tmp)
        #print()
        print(tmpDiffAngle)
        readSamples1(1,tmpDiffAngle)
        #readSamples1(1,angle1)
        print("diffSample:",sample_1[0])
        #newY.append((newY[len(newY)-1]+tmpGoal+sample_1[0])/2.0+sample_1[0]/5)
        if tmpGoal>oldY[len(oldY)-1]:
            newY.append(tmpGoal-sample_1[0])
        else:
            newY.append(tmpGoal+sample_1[0])
        #newY.append(tmpGoal+sample_1[0])
        
        #tmpInterval=sampleDeltaX[i]/4.0
        midP1=midPointDisplacement(newX[len(newX)-2],newX[len(newX)-1],newY[len(newY)-2],newY[len(newY)-1],readSamples3()/4)
        midP2=midPointDisplacement(newX[len(newX)-2],midP1[0],newY[len(newY)-2],midP1[1],readSamples3()/8)
        midP3=midPointDisplacement(midP1[0],newX[len(newX)-1],midP1[1],newY[len(newY)-1],readSamples3()/8)
        
        midP4=midPointDisplacement(newX[len(newX)-2],midP2[0],newY[len(newY)-2],midP2[1],readSamples3()/16)
        midP5=midPointDisplacement(midP2[0],midP1[0],midP2[1],midP1[1],readSamples3()/16)
        midP6=midPointDisplacement(midP1[0],midP3[0],midP1[1],midP3[1],readSamples3()/16)
        midP7=midPointDisplacement(midP3[0],newX[len(newX)-1],midP3[1],newY[len(newY)-1],readSamples3()/16)
        
        newX.insert(len(newX)-1,midP4[0])
        newX.insert(len(newX)-1,midP2[0])
        newX.insert(len(newX)-1,midP5[0])
        newX.insert(len(newX)-1,midP1[0])
        newX.insert(len(newX)-1,midP6[0])
        newX.insert(len(newX)-1,midP3[0])
        newX.insert(len(newX)-1,midP7[0])
        
        newY.insert(len(newY)-1,midP4[1])
        newY.insert(len(newY)-1,midP2[1])
        newY.insert(len(newY)-1,midP5[1])
        newY.insert(len(newY)-1,midP1[1])
        newY.insert(len(newY)-1,midP6[1])
        newY.insert(len(newY)-1,midP3[1])
        newY.insert(len(newY)-1,midP7[1])
        
       
    canvas1.delete("synthesizedSketch")
    for i in range(1,len(newX)):
        canvas1.create_line(newX[i], newY[i], newX[i-1], newY[i-1], fill='green',tags='synthesizedSketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)
def nextSythesis1():

    global cubicSpline_sketch
    cubicSpline_sketch=UnivariateSpline(np.array(tmpX_sketch),np.array(tmpY_sketch))
    cubicSpline_sketch.set_smoothing_factor(0)
    
    global sampleDeltaX
    sampleDeltaX=[]
    minX=tmpX_sketch[0]
    maxX=tmpX_sketch[len(tmpX_sketch)-1]
    tmpCount=minX
    while(1):
        readSamples2(1)
        if tmpCount+sample_2[0]<maxX:
            sampleDeltaX.append(sample_2[0])
            tmpCount+=sample_2[0]
        #if tmpCount+sample_2[0]==maxX:
        else:
            sampleDeltaX.append(maxX-tmpCount)
            #tmpCount+=sample_2[0]
            break
    global newX,newY 
    newX=[tmpX_sketch[0]]
    newY=[tmpY_sketch[0]+shift]
    #oldX=[sketchX[index-1][0]]
    oldY=[tmpY_sketch[0]+shift]
    for i in range(len(sampleDeltaX)):
        #newX.append((newX[len(newX)-1]+newX[len(newX)-1]+sampleDeltaX[i])/2.0)
        newX.append(newX[len(newX)-1]+sampleDeltaX[i])
        tmpGoal=float(cubicSpline_data(newX[len(newX)-1]))
        oldY.append(float(cubicSpline_sketch(newX[len(newX)-1]))+shift)
        #angle1=math.atan2(tmpGoal-newY[len(newY)-1], sampleDeltaX[i])
        #angle2=math.atan2(oldY[len(oldY)-1]-oldY[len(oldY)-2], sampleDeltaX[i])
        
        tmpx1=np.array([sampleDeltaX[i],tmpGoal-newY[len(newY)-1]])
        
        #tmpx2=np.array([sampleDeltaX[i],oldY[len(oldY)-1]-oldY[len(oldY)-2]])
        tmpx2=np.array([sampleDeltaX[i],oldY[len(oldY)-1]-newY[len(newY)-1]])
        #print((np.sqrt(tmpx1.dot(tmpx1))*np.sqrt(tmpx2.dot(tmpx2))))
        #tmpDiffAngle=abs(angle1-angle2)
        tmp=tmpx1.dot(tmpx2)/(np.sqrt(tmpx1.dot(tmpx1))*np.sqrt(tmpx2.dot(tmpx2)))
        if tmp>1:
            tmp=1
        if tmp<-1:
            tmp=-1
        tmpDiffAngle=np.arccos(tmp)
        #print()
        print(tmpDiffAngle)
        readSamples1(1,tmpDiffAngle)
        #readSamples1(1,angle1)
        print("diffSample:",sample_1[0])
        #newY.append((newY[len(newY)-1]+tmpGoal+sample_1[0])/2.0+sample_1[0]/5)
        if tmpGoal>oldY[len(oldY)-1]:
            newY.append(tmpGoal-sample_1[0])
        else:
            newY.append(tmpGoal+sample_1[0])
        #newY.append(tmpGoal+sample_1[0])
        
        #tmpInterval=sampleDeltaX[i]/4.0
        midP1=midPointDisplacement(newX[len(newX)-2],newX[len(newX)-1],newY[len(newY)-2],newY[len(newY)-1],readSamples3()/4)
        midP2=midPointDisplacement(newX[len(newX)-2],midP1[0],newY[len(newY)-2],midP1[1],readSamples3()/8)
        midP3=midPointDisplacement(midP1[0],newX[len(newX)-1],midP1[1],newY[len(newY)-1],readSamples3()/8)
        
        midP4=midPointDisplacement(newX[len(newX)-2],midP2[0],newY[len(newY)-2],midP2[1],readSamples3()/16)
        midP5=midPointDisplacement(midP2[0],midP1[0],midP2[1],midP1[1],readSamples3()/16)
        midP6=midPointDisplacement(midP1[0],midP3[0],midP1[1],midP3[1],readSamples3()/16)
        midP7=midPointDisplacement(midP3[0],newX[len(newX)-1],midP3[1],newY[len(newY)-1],readSamples3()/16)
        
        newX.insert(len(newX)-1,midP4[0])
        newX.insert(len(newX)-1,midP2[0])
        newX.insert(len(newX)-1,midP5[0])
        newX.insert(len(newX)-1,midP1[0])
        newX.insert(len(newX)-1,midP6[0])
        newX.insert(len(newX)-1,midP3[0])
        newX.insert(len(newX)-1,midP7[0])
        
        newY.insert(len(newY)-1,midP4[1])
        newY.insert(len(newY)-1,midP2[1])
        newY.insert(len(newY)-1,midP5[1])
        newY.insert(len(newY)-1,midP1[1])
        newY.insert(len(newY)-1,midP6[1])
        newY.insert(len(newY)-1,midP3[1])
        newY.insert(len(newY)-1,midP7[1])
        
       
#    canvas1.delete("synthesizedSketch")
#    for i in range(1,len(newX)):
#        canvas1.create_line(newX[i], newY[i], newX[i-1], newY[i-1], fill='green',tags='synthesizedSketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)
##7F00FF
midPoint=[]
def midPointDisplacement(newX1,newX2,newY1,newY2,displacement):
    global midPoint
    midPoint=[]
    midPoint.append((newX1+newX2)/2.0)
    if cubicSpline_sketch(midPoint[0])>=(newY1+newY2)/2.0:
        midPoint.append((newY1+newY2)/2.0+displacement)
    else:
        midPoint.append((newY1+newY2)/2.0-displacement)
    return midPoint
#    iteration+=1
#    if iteration==2:
#        return
#    midPointDisplacement(newX1,midPointX,newY1,midPointY,displacement/10)
#    midPointDisplacement(midPointX,newX2,midPointY,newY2,displacement/10)
    
#
window = tk.Tk()
window.title('my window')
window.geometry('920x450+500+0')
canvasWidth1=900
canvasHeight1=400
canvas1 = tk.Canvas(window, bg='white', height=canvasHeight1, width=canvasWidth1)
canvas1.pack()
frame1 = tk.Frame(window)
var1 = tk.StringVar()
btNext = tk.Button(frame1, text ="Next",command=nextUserStudyData)
btNextSythesis = tk.Button(frame1, text ="NextSythesis",command=nextSythesis)
lbNum = tk.Label(frame1,textvariable=var1, bg='red',font=('Arial', 12), width=5)
btNext.grid(row=1,column=1)
btNextSythesis.grid(row=1,column=2)
lbNum.grid(row=1,column=3)
frame1.pack()
leftSketchPanel=350
rightSketchPanel=550
offsetH=100
sketchPanelColor='#8CF2C2'
canvas1.create_line(leftSketchPanel, 0+offsetH, leftSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
canvas1.create_line(leftSketchPanel, 0+offsetH, rightSketchPanel, 0+offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
canvas1.create_line(rightSketchPanel, 0+offsetH, rightSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5)) 
canvas1.create_line(leftSketchPanel, canvasHeight1-offsetH, rightSketchPanel, canvasHeight1-offsetH, fill=sketchPanelColor,tags='sketchPanel',joinstyle=tk.ROUND, width=1.5,dash=(3,5))
#train_x, valid_x, train_y, valid_y, train_z,valid_z = train_test_split(angle1, absDiff, l1, test_size=0.33, random_state = 1)
#train_x, valid_x, train_y, valid_y, train_z,valid_z = train_test_split(angle1, diff, l1, test_size=0, random_state = 1)
train_x, valid_x, train_y, valid_y, train_z,valid_z = train_test_split(diffAngle, absDiff, l1, test_size=0, random_state = 1)
#plt.scatter(train_x, train_y, facecolor='None', edgecolor='k', alpha=0.3)
#plt.show()
#len(train_y)
#initialLength=len(train_x)
#for i in range(initialLength-1,-1,-1):
#    if train_y[i]>20:
#        #print(i)
#        train_y=train_y.drop(train_y.index[0])
        #train_x=train_x.drop(train_x.index[i])
    #print(i)

weights = np.polyfit(train_x, train_y,3)
model1 = np.poly1d(weights)

#pred = model1(valid_x)
#plt.scatter(valid_x, valid_y, facecolor='None', edgecolor='k', alpha=0.3)
#xp = np.linspace(valid_x.min(),valid_x.max(),70)
#pred_plot = model1(xp)
#plt.plot(xp, pred_plot)
#plt.show()

pred = model1(train_x)
plt.scatter(train_x, train_y, facecolor='None', edgecolor='k', alpha=0.3)
xp = np.linspace(train_x.min(),train_x.max(),70)
pred_plot = model1(xp)
#pred2=1/(20735390.331*xp+20736063.388)+4.85463713
plt.plot(xp, pred_plot)
#plt.plot(xp, pred2)
#plt.show()
#r2_score(pred,train_y)

absFirst=abs(pred-train_y)

weights2 = np.polyfit(train_x, absFirst,3)
model2 = np.poly1d(weights2)
pred_plot2=model2(xp)
plt.scatter(train_x, absFirst, facecolor='None', edgecolor='k', alpha=0.3)
plt.plot(xp, pred_plot2)
#plt.show()
#r2_score(model2(train_x),absFirst)
#histabsFirst=plt.hist(absFirst)
#histAngle=plt.hist(angle1)
#plt.show()
#histDiff=plt.hist(diff)
#plt.show()



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
#loadUserStudyData2()

def loadUserStudyData1():
    global selectedData,sketchX,sketchY
    df2 = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy.csv')
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
        tmpSketchY.append(row['sketch'])
        tmp=tmpSketchY[0][1:len(tmpSketchY[0])].split(";")
        tmp_SketchY=[]
        for i in range(0,len(tmp)):
            tmp_SketchY.append(float(tmp[i]))
        sketchY.append(tmp_SketchY)
loadUserStudyData1()



window.mainloop()

sample_1=[]
sample_2=[]
xmin1=0
xmax1=8
#xmin1=0
#xmax1=3
print(xmax1)
def readSamples1(samplesNum,angle):
    global sample_1
    sample_1=[]

    i1=0
    print("angle",angle)
    if angle>0.8:
        u=8
        std=1
    else:
        u=model1(angle)
        std=model2(angle)
        
#    print("u",u)
#    print("std",std)
        
    c1=normalDistribution(u,u,std)
    while i1<samplesNum:
#        print("angle",angle)
#        print("u",u)
#        print("std",std)
        Y=random.uniform(xmin1, xmax1)
        U=random.random()
        if U<=normalDistribution(Y, u, std)/c1:
            sample_1.append(Y)
            i1+=1
        #print("1")
#def beta(x,alpha1,alpha2,a,b):
#    return gamma(alpha1+alpha2)/(gamma(alpha1)*gamma(alpha2))*(math.pow(x-a,alpha1-1)*math.pow(b-x,alpha2-x))/math.pow(b-a,alpha1+alpha2-1)
def logarithmic(x,theta):
    return (-theta**x/(x*math.log(1-theta)))
def normalDistribution(x,u,std):
    return math.pow(math.e,-math.pow((x-u)/std,2)/2.0)/(std*math.sqrt(2*math.pi))
    
#theta=0.30244 #interval=1
#theta=0.67395 #interval=4
#theta=0.33104 #interval=5
#theta=0.81901 #interval=8
#theta=0.85249 #interval=10
theta=0.88563 #interval=12
c2=logarithmic(1,theta)
xmin2=min(deltaXNum)
xmax2=max(deltaXNum)
def readSamples2(samplesNum):
    i2=0
    global sample_2
    sample_2=[]
    while i2<samplesNum:        
        Y=random.randint(xmin2, xmax2)
        U=random.random()
        if U<=logarithmic(Y, theta)/c2:
            sample_2.append(Y+offset)
            i2+=1
        #print(2)
#readSamples2(5000)

def readSamples3():
    u=2.36
    std=3.78
    c3=normalDistribution(u,u,std)
    while True:
        #print(3)
        Y=random.uniform(0, 10)
        U=random.random()
        if U<=normalDistribution(Y, u, std)/c3:
            break
    return Y

newData=[]

def save(newSketchX,newSketchY):
    newSketchData=[]
    tmpNewSketchX="*"
    for i in range(0,len(newSketchX)):
        tmpNewSketchX+=str(newSketchX[i])+";"
    tmpNewSketchY="*"
    for i in range(0,len(newSketchY)):
        tmpNewSketchY+=str(newSketchY[i])+";"
    newSketchData.append(tmpNewSketchX)
    newSketchData.append(tmpNewSketchY)
    with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/sketchSynthesis.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(newSketchData)


def generateSketches():
    global newData
    for i in range(300):
        nextUserStudyData1()
        if index in array:
            newData.append([tmpX_sketch,tmpY_sketch])
            save(tmpX_sketch,tmpY_sketch)
            for j in range(4):
                nextSythesis1()
                newData.append([newX,newY])
                save(newX,newY)
        else:
            
            for j in range(20):
                newData.append([tmpX_sketch,tmpY_sketch])
                save(tmpX_sketch,tmpY_sketch)
                for m in range(4):
                    nextSythesis1()
                    newData.append([newX,newY])
                    save(newX,newY)

    
    print("1")

#generateSketches()
#x=[]
#y=[]
#z=[]
#x2=[]
#y2=[]
#x3=[]
#x4=[]
#len(l1)
#for i in range(120000,150000):
#    x.append(l1[i])
#    y.append(angle1[i])
#    z.append(diff[i])
#    x2.append(l2[i])
#    y2.append(angle2[i])
#    x3.append(deltaX[i])
#    x4.append(angle3[i])
#p=[[6],[7],[3]]

#fig=plt.figure()
#ax=fig.add_subplot(111,projection='3d')
#ax=Axes3D(fig)
#ax.scatter(x4,z)
#ax.scatter(x,y,z,c='r',makrer='0')
#ax.set_xlabel('x axis')
#ax.set_ylabel('y axis')
#ax.set_ylabel('y axis')
#plt.scatter(x2,z)
#plt.show()
#model=LinearRegression()
#model.fit(x,y)
#test=[[1,2]]
#predictions=model.predict(test)

    


