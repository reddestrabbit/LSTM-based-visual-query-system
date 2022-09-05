# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 17:02:05 2020

@author: Chaoran Fan
"""
import pandas as pd
import tkinter as tk
import numpy as np
from pandas import read_csv
import math
from scipy.interpolate import UnivariateSpline
import random
import csv
from PIL import Image, ImageDraw
#white=(255,255,255)
#black=(0,0,0)
#image1 = Image.new("RGB", (200, 200), black)
#draw = ImageDraw.Draw(image1)
#green = (0,128,0)
#draw.line([0, 100, 100, 100], green)
## PIL image can be saved as .png .jpg .gif or .bmp file (among others)
#filename = "my_drawing.jpg"
#image1.save(filename)

df = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy.csv')

userId=[]
count=[]
sketchNum=[]
comparePoint=[]
startPoint=[]
sketchLength=[]
interval=1
selectedTraining=[]
sketchTraining=[]
simTraining=[]

def loadUserStudyData():
    global selectedTraining,sketchTraining,simTraining
    global trainingSize
    global dataSetNum,dataScale
    trainingSize=df.shape[0]
    dataSetNum=df['dataSet']
    dataSetNum=dataSetNum.values
    dataScale=df['scale']
    dataScale=dataScale.values
    simTraining=df['sim']
    
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

def draw():
    window = tk.Tk()
    window.title('my window')
    window.geometry('920x650+500+0')

    canvas1 = tk.Canvas(window, bg='white', height=600, width=900)
    canvas1.pack()
    xSamples=np.linspace(350,550,201)
    
    for index in range(len(userId)):
        image1 = Image.new("RGB", (900, 600), 'white')
        draw = ImageDraw.Draw(image1)
        originalPoints2=np.c_[xSamples,selectedTraining[index][100:301]]
        for i in range(1,originalPoints2.shape[0]):
            if sketchTraining[index][i]==-1 or sketchTraining[index][i-1]==-1:
                #print(originalPoints2[i][0])
                if originalPoints2[i][0]<500:
                    canvas1.create_line(originalPoints2[i][0], originalPoints2[i][1], originalPoints2[i-1][0], originalPoints2[i-1][1], fill='red',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
            else:
                canvas1.create_line(originalPoints2[i][0], originalPoints2[i][1], originalPoints2[i-1][0], originalPoints2[i-1][1], fill='orange',tags='curve',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5)
                draw.line([originalPoints2[i][0], originalPoints2[i][1], originalPoints2[i-1][0], originalPoints2[i-1][1]], fill="orange",width=6)
        
        sketchPoints=np.c_[xSamples,sketchTraining[index]]
        for i in range(1,sketchPoints.shape[0]):
            if sketchPoints[i][1]!=-1 and sketchPoints[i-1][1]!=-1:
                canvas1.create_line(sketchPoints[i][0], sketchPoints[i][1], sketchPoints[i-1][0], sketchPoints[i-1][1], fill='purple',tags='sketch',joinstyle=tk.ROUND, capstyle=tk.ROUND,width=5.5) 
                draw.line([sketchPoints[i][0], sketchPoints[i][1], sketchPoints[i-1][0], sketchPoints[i-1][1]], fill="purple",width=6)
        fileName="C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/user study1 pics/"+str(userId[index])+"_"+str(sketchNum[index])+"_"+str(count[index])+"_"+str(simTraining[index])+".png"
        image1.save(fileName)
        print(fileName)

    #window.mainloop()
draw()




