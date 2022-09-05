# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:19:18 2019

@author: Chaoran Fan
"""
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Lambda, Flatten, TimeDistributed
from keras.layers import LSTM, Dense
from keras.models import load_model
import keras.backend as K
from scipy.interpolate import UnivariateSpline
import pandas as pd
hunits = 20
def exponent_neg_manhattan_distance(left, right):
    ''' Helper function for the similarity estimate of the LSTMs outputs'''
    return K.exp(-K.sum(K.abs(left-right), axis=-1, keepdims=True))

model = load_model('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/SiameseLSTM.h5',custom_objects={'exponent_neg_manhattan_distance': exponent_neg_manhattan_distance})
model.summary()
inp = Input(batch_shape= (None,None, 1))
rnn= LSTM(hunits, return_sequences=True, name="RNN")(inp)
states = Model(inputs=[inp],outputs=[rnn])
states.summary()
states.layers[1].set_weights(model.layers[2].get_weights())   
df = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudy2.csv')
df3 = pd.read_csv('C:/Users/Chaoran Fan/Dropbox/interaction in visual analytics/sketch query/data/userStudyForTraining - Copy2.csv')
expanded=20
newTailY=[]
realTail=[]
import numpy as np
import math
def loadNewTail():
    global newTailY,realTail
    newTailY=[]
    realTail=[]
    for rowId, row in df3.iterrows():
        tmpSelected=[]
        tmpSelected.append(row['diff'])
            
        tmp=tmpSelected[0][1:len(tmpSelected[0])-1].split(";")
        y=[]
        
        for i in range(0,len(tmp)):
            y.append(float(tmp[i]))
        newTailY.append(y)
        
        if len(realTail)==0:
            tmpSelected=[]
            tmpSelected.append(row['realTail'])
            tmp=tmpSelected[0][1:len(tmpSelected[0])-1].split(";")
            y=[]
        
            for i in range(0,len(tmp)):
                y.append(float(tmp[i]))
            realTail.append(y)
        

loadNewTail()
selectedTraining=[]
ref=[]
def loadUserStudyData():
    global selectedTraining,ref

    index=0
    for rowId, row in df.iterrows():
        tmpSelected=[]
        tmpSelected.append(row['selected'])
        tmpSketch=[]
        tmpSketch.append(row['sketch'])
        tmpSketchX=[]
        tmpSketchX.append(row['sketch_x'])

        
        tmp=tmpSketchX[0].split(";")
        tmpSketchXTraining=[]
        for i in range(0,len(tmp)):
            tmpSketchXTraining.append(float(tmp[i]))
        
        tmpSketchXTraining=np.array(tmpSketchXTraining)
        
       
        tmp_xSamples_sketch=np.linspace(tmpSketchXTraining[0],tmpSketchXTraining[len(tmpSketchXTraining)-1], 41)
        
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
            
        for i in range(0,len(realTail[0])):
            ref.append(realTail[0][i])
        ref.extend(tmpselectedTraining3)
        for j in range(0,expanded):
            tmpselectedTraining=[]
            for i in range(0,len(newTailY[0])):
                tmpselectedTraining.append(newTailY[index*expanded+j][i])
            
            tmpselectedTraining.extend(tmpselectedTraining3)
            selectedTraining.append(tmpselectedTraining)
            
        index+=1
loadUserStudyData()

def computeCosSim(a,b):
    dot = np.dot(a, b)
    norma = np.linalg.norm(a)
    normb = np.linalg.norm(b)
    cos = dot / (norma * normb)
    return cos

rnsALL=[]
rnsRef=[]
#sim=[]

ref=np.array(ref).reshape(-1,len(np.array(ref)),1)
rnsRef=states.predict(ref)
#c=

for i in range(len(selectedTraining)):
    tmp=np.array(selectedTraining[i]).reshape(-1,len(np.array(selectedTraining[i])),1)
    rns = states.predict(tmp)
    
#    for j in range(20,59):
#        sim.append(computeCosSim(rns[0][j],rnsRef[0][j]))
#    
#    sim.append(computeCosSim(rns[0][20],rnsRef[0][20]))
    rnsALL.append(rns)
simALL=[]
for i in range(0,60):
    sim=[]
    for j in range(len(rnsALL)):
        sim.append(computeCosSim(rnsALL[j][0][i],rnsRef[0][i]))
    sim.sort()
    simALL.append(sim)
iqr=[]
for i in range(len(simALL)):
    iqr.append(simALL[i][int(len(simALL[i])*0.75)]-simALL[i][int(len(simALL[i])*0.25)])

#sim.sort()
#sim[int(len(sim)*0.75)]-sim[int(len(sim)*0.25)]


#lastALL=[]
#for i in range(30):
#    lastALL.append(rnsALL[i][0][58])
#meanALL=[]
#
#for i in range(20):
#    meanALL.append(0)
#for i in range(30):
#    for j in range(20):
#        meanALL[j]+=lastALL[i][j]
#for i in range(20):
#    meanALL[i]/=20
#variance=[]
#for i in range(30):
#    tmp=0
#    for j in range(20):
#        tmp+=(lastALL[i][j]-meanALL[j])**2
#    variance.append(tmp/20)

