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
import math

def save():
    print("saved")

selectedData=[]
sketchX=[]
sketchY=[]
sketchNum=[]
userId=[]
count=[]
def loadUserStudyData2():
    global selectedData,sketchX,sketchY,userId,count,sketchNum
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
    userId=df2['userId']
    sketchNum=df2['sketchNum']
    count=df2['count']
loadUserStudyData2()

table=[]
xSamples=np.linspace(350,550,201)
interval=12
for m in range(0,len(sketchNum)):
    if m%12==0:
        tmp_cubicSpline_data= UnivariateSpline(xSamples,np.array(selectedData[m]))
        tmp_cubicSpline_data.set_smoothing_factor(0)
    tmp_cubicSpline_sketch= UnivariateSpline(np.array(sketchX[m]),np.array(sketchY[m]))
    tmp_cubicSpline_sketch.set_smoothing_factor(0)
    shift=np.mean(selectedData[m])-np.mean(sketchY[m])
    for n in range(0,len(sketchX[m])-interval,interval):
        midDiff=abs((sketchY[m][n+interval]+sketchY[m][n])/2.0-tmp_cubicSpline_sketch((sketchX[m][n+interval]+sketchX[m][n])/2.0))
#        if n==0:
#            shift=float(tmp_cubicSpline_data(sketchX[m][n]))-sketchY[m][n]
        tmpId=int(m/120)+1
        tmpCount=int(m/12)+1-10*(tmpId-1)
        tmpGoal=float(tmp_cubicSpline_data(sketchX[m][n+interval]))
        delta=sketchX[m][n+interval]-sketchX[m][n]
        L1=math.sqrt((delta)**2+(tmpGoal-(sketchY[m][n]+shift))**2)
        L2=math.sqrt((delta)**2+(sketchY[m][n+interval]-sketchY[m][n])**2)
        tmpx1=np.array([delta,tmpGoal-sketchY[m][n]])
        tmpx2=np.array([delta,sketchY[m][n+interval]-sketchY[m][n]])
        #tmpy=np.array([1,0])     
        tmpy=np.array([0,1]) 
        angle3=np.arccos(tmpx1.dot(tmpy)/(np.sqrt(tmpx1.dot(tmpx1))*np.sqrt(tmpy.dot(tmpy))))
        angle1=math.atan2(tmpGoal-sketchY[m][n], delta)
        #angle2=np.arccos(tmpx2.dot(tmpy)/(np.sqrt(tmpx2.dot(tmpx2))*np.sqrt(tmpy.dot(tmpy))))
        angle2=math.atan2(sketchY[m][n+interval]-sketchY[m][n], delta)
        diff=sketchY[m][n+interval]-tmpGoal
        #diffAngle=abs(angle1-angle2)
        diffAngle=np.arccos(tmpx1.dot(tmpx2)/(np.sqrt(tmpx1.dot(tmpx1))*np.sqrt(tmpx2.dot(tmpx2))))
        if (m+1)%12==0:
            table.append([tmpId,tmpCount,12,sketchX[m][n],sketchY[m][n]+shift,sketchX[m][n+interval],sketchY[m][n+interval]+shift,delta,tmpGoal,L1,L2,angle1,angle2,angle3,diff,abs(diff),diffAngle,interval,midDiff])
        else:
            table.append([tmpId,tmpCount,(m+1)%12,sketchX[m][n],sketchY[m][n]+shift,sketchX[m][n+interval],sketchY[m][n+interval]+shift,delta,tmpGoal,L1,L2,angle1,angle2,angle3,diff,abs(diff),diffAngle,interval,midDiff])

with open('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudyTable.csv', 'a', newline='') as f:
    writer = csv.writer(f)
    for i in range(len(table)):
        writer.writerow(table[i])

    
window = tk.Tk()
window.title('my window')
window.geometry('920x450+500+0')

canvasWidth1=900
canvasHeight1=400
canvas1 = tk.Canvas(window, bg='white', height=canvasHeight1, width=canvasWidth1)
canvas1.pack()

window.mainloop()