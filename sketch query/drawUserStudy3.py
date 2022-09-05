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

df = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy3New.csv')

userId=[]
count=[]
sketch_x=[]
sketch_y=[]
results1X=[]
results1Y=[]
results2X=[]
results2Y=[]
results3X=[]
results3Y=[]
our=[]
qetch=[]
dtw=[]

def draw(rowId):
    
    image1 = Image.new("RGB", (600, 400), 'white')#width height
    draw = ImageDraw.Draw(image1)
    for i in range(1,len(sketch_x)):
        draw.line([sketch_x[i]-350, sketch_y[i], sketch_x[i-1]-350, sketch_y[i-1]], fill="purple",width=6)
    for i in range(1,len(sketch_x)):
        draw.line([sketch_x[i]-350+200, sketch_y[i], sketch_x[i-1]-350+200, sketch_y[i-1]], fill="purple",width=6)
    for i in range(1,len(sketch_x)):
        draw.line([sketch_x[i]-350+200+200, sketch_y[i], sketch_x[i-1]-350+200+200, sketch_y[i-1]], fill="purple",width=6)
    for i in range(1,len(results1X)):
        draw.line([results1X[i]-350, results1Y[i], results1X[i-1]-350, results1Y[i-1]], fill="red",width=6)
    for i in range(1,len(results2X)):
        draw.line([results2X[i]-350+200, results2Y[i], results2X[i-1]-350+200, results2Y[i-1]], fill="green",width=6)
    for i in range(1,len(results3X)):
        draw.line([results3X[i]-350+200+200, results3Y[i], results3X[i-1]-350+200+200, results3Y[i-1]], fill="blue",width=6)
        
    fileName="C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/user study3 pics/"+str(userId[rowId])+"_"+str(count[rowId])+"_our-"+str(our[rowId])+"_qetch-"+str(qetch[rowId])+"_dtw-"+str(dtw[rowId])+".png"
    image1.save(fileName)
    print(fileName)

def loadUserStudyData():
    
    global userId, count, sketch_x, sketch_y,results1X,results1Y,results2X,results2Y,results3X,results3Y,our,qetch,dtw
    userId=df['userId']
    count=df['count']
    our=df['our']
    qetch=df['qetch']
    dtw=df['dtw']
#    sketch_x=df['sketch_x']
#    sketch_y=df['sketch_y']
#    results1X=df['results1X']
#    results1Y=df['results1Y']
#    results2X=df['results2X']
#    results2Y=df['results2Y']
#    results3X=df['results3X']
#    results3Y=df['results3Y']
    for rowId, row in df.iterrows():
        tmpSketchX=[]
        tmpSketchX.append(row['sketch_x'])
        tmpSketchY=[]
        tmpSketchY.append(row['sketch_y'])
        tmp1x=[]
        tmp1x.append(row['results1X'])
        tmp1y=[]
        tmp1y.append(row['results1Y'])
        tmp2x=[]
        tmp2x.append(row['results2X'])
        tmp2y=[]
        tmp2y.append(row['results2Y'])
        tmp3x=[]
        tmp3x.append(row['results3X'])
        tmp3y=[]
        tmp3y.append(row['results3Y'])
        
        sketch_x=[]
        sketch_y=[]
        results1X=[]
        results1Y=[]
        results2X=[]
        results2Y=[]
        results3X=[]
        results3Y=[]
        
        tmp=tmpSketchX[0].split(";")
        for i in range(len(tmp)):
            sketch_x.append(float(tmp[i]))
                
        tmp=tmpSketchY[0].split(";")
        for i in range(len(tmp)):
            sketch_y.append(float(tmp[i]))
        
        tmp=tmp1x[0].split(";")
        for i in range(len(tmp)):
            results1X.append(float(tmp[i]))
        
        tmp=tmp1y[0].split(";")
        for i in range(len(tmp)):
            results1Y.append(float(tmp[i]))
        
        tmp=tmp2x[0].split(";")
        for i in range(len(tmp)):
            results2X.append(float(tmp[i]))
        
        tmp=tmp2y[0].split(";")
        for i in range(len(tmp)):
            results2Y.append(float(tmp[i]))
            
        tmp=tmp3x[0].split(";")
        for i in range(len(tmp)):
            results3X.append(float(tmp[i]))
        
        tmp=tmp3y[0].split(";")
        for i in range(len(tmp)):
            results3Y.append(float(tmp[i]))

#        print(rowId)
        draw(rowId)
            



loadUserStudyData()



    #window.mainloop()
#draw()




