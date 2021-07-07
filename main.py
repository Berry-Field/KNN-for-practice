# -*- coding: utf-8 -*-
"""
Created on Sun Fir 2 10:07:29 2021
@author: JIA HB
&&&Reference marked&&&
"""
from tkinter import *
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import operator

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(sortedClassCount)
    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename,'r',encoding = 'utf-8')
    arrayOLines = fr.readlines()
    arrayOLines[0]=arrayOLines[0].lstrip('\ufeff')
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classLabelVector = []
    index = 0

    for line in arrayOLines:
        line = line.strip()   
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if listFromLine[-1] == 'UUVs':
            classLabelVector.append(1)
        elif listFromLine[-1] == 'animals':
            classLabelVector.append(2)
        elif listFromLine[-1] == 'stone':
            classLabelVector.append(3)
        index += 1
    return returnMat, classLabelVector    

def showdatas(datingDataMat, datingLabels):
    font = FontProperties(fname=r"c:\windows\fonts\msyh.ttc", size=14) 
    fig, axs = plt.subplots(nrows=2, ncols=2,sharex=False, sharey=False, figsize=(13,8))
    numberOfLabels = len(datingLabels)
    LabelsColors = []
    for i in datingLabels:
        if i == 1:
            LabelsColors.append('black')
        if i == 2:
            LabelsColors.append('orange')
        if i == 3:
            LabelsColors.append('red')
    axs[0][0].scatter(x=datingDataMat[:,0], y=datingDataMat[:,1], color=LabelsColors,s=15, alpha=.5)
    axs0_title_text = axs[0][0].set_title(u'数据参数1与数据参数2',FontProperties=font)
    axs0_xlabel_text = axs[0][0].set_xlabel(u'数据参数1',FontProperties=font)
    axs0_ylabel_text = axs[0][0].set_ylabel(u'数据参数2',FontProperties=font)
    plt.setp(axs0_title_text, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel_text, size=7, weight='bold', color='black')

    axs[0][1].scatter(x=datingDataMat[:,0], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    axs1_title_text = axs[0][1].set_title(u'数据参数1与数据参数3',FontProperties=font)
    axs1_xlabel_text = axs[0][1].set_xlabel(u'数据参数1',FontProperties=font)
    axs1_ylabel_text = axs[0][1].set_ylabel(u'数据参数3',FontProperties=font)
    plt.setp(axs1_title_text, size=9, weight='bold', color='red')
    plt.setp(axs1_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs1_ylabel_text, size=7, weight='bold', color='black')

    axs[1][0].scatter(x=datingDataMat[:,1], y=datingDataMat[:,2], color=LabelsColors,s=15, alpha=.5)
    axs2_title_text = axs[1][0].set_title(u'数据参数2与数据参数3',FontProperties=font)
    axs2_xlabel_text = axs[1][0].set_xlabel(u'数据参数2',FontProperties=font)
    axs2_ylabel_text = axs[1][0].set_ylabel(u'数据参数3',FontProperties=font)
    plt.setp(axs2_title_text, size=9, weight='bold', color='red')
    plt.setp(axs2_xlabel_text, size=7, weight='bold', color='black')
    plt.setp(axs2_ylabel_text, size=7, weight='bold', color='black')

    UUVs = mlines.Line2D([], [], color='black', marker='.',
                      markersize=6, label='UUVs')
    animals = mlines.Line2D([], [], color='orange', marker='.',
                      markersize=6, label='animals')
    stone = mlines.Line2D([], [], color='red', marker='.',
                      markersize=6, label='stone')
    axs[0][0].legend(handles=[UUVs,animals,stone])
    axs[0][1].legend(handles=[UUVs,animals,stone])
    axs[1][0].legend(handles=[UUVs,animals,stone])

    plt.show()    


def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    rate = 0.10
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * rate)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:],
            datingLabels[numTestVecs:m], 4)
        print("分类结果:%s\t真实类别:%d" % (classifierResult, datingLabels[i]))
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print("错误率:%f%%" %(errorCount/float(numTestVecs)*100))    

def classifyThings():
    resultList = ['UUVs','animals','stone']
    dataone = float(input("测量参数1:"))
    datawtwo = float(input("测量参数2:"))
    datathree = float(input("测量参数3:"))
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([datawtwo, dataone, datathree])
    norminArr = (inArr - minVals) / ranges
    classifierResult = classify0(norminArr, normMat, datingLabels, 3)
    print("目标可能是" % (resultList[classifierResult-1]))    


class core():
    def __init__(self, str):
        self.string = str



    def classifytarget(self):
        resultList = ['UUVs','animals','stone']
        precentTats = float(input("测量参数1:"))
        ffMiles = float(input("测量参数2:"))
        iceCream = float(input("测量参数3:"))
        filename = "datingTestSet.txt"
        datingDataMat, datingLabels = file2matrix(filename)
        normMat, ranges, minVals = autoNorm(datingDataMat)
        inArr = np.array([ffMiles, precentTats, iceCream])
        norminArr = (inArr - minVals) / ranges
        classifierResult = classify0(norminArr, normMat, datingLabels, 3)
        print("目标可能是" % (resultList[classifierResult-1]))


    def knn_predict(self, string):
        resultList = ['UUVs','animals','stone']
        [precentTats,ffMiles, iceCream]=(string.split(',',2))
        precentTats=float(precentTats)
        ffMiles = float(ffMiles)
        iceCream = float(iceCream)
        filename = "datingTestSet.txt"
        datingDataMat, datingLabels = file2matrix(filename)
        normMat, ranges, minVals = autoNorm(datingDataMat)
        inArr = np.array([ffMiles, precentTats, iceCream])
        norminArr = (inArr - minVals) / ranges
        classifierResult = classify0(norminArr, normMat, datingLabels, 3)
        return resultList[classifierResult-1]

    def main(self):
        s = self.knn_predict(self.string)
        return s    

root = Tk()
root.title("目标分析") 
sw = root.winfo_screenwidth()
sh = root.winfo_screenheight()
ww = 500
wh = 300
x = (sw - ww) / 2
y = (sh - wh) / 2 - 50
root.geometry("%dx%d+%d+%d" % (ww, wh, x, y)) 

lb2 = Label(root, text="输入内容，按回车键分析")
lb2.place(relx=0, rely=0.05)

txt = Text(root, font=("宋体", 20))
txt.place(rely=0.7, relheight=0.3, relwidth=1)

inp1 = Text(root, height=15, width=65, font=("宋体", 18))
inp1.place(relx=0, rely=0.2, relwidth=1, relheight=0.4)


def run1():
    txt.delete("0.0", END)
    a = inp1.get('0.0', (END))
    p = core(a)
    s = p.main()
    print(s)
    txt.insert(END, s) 


def button1(event):
    btn1 = Button(root, text='分析', font=("", 12), command=run1)  
    btn1.place(relx=0.35, rely=0.6, relwidth=0.15, relheight=0.1)

if __name__ == '__main__':
    filename = "datingTestSet.txt"
    datingDataMat, datingLabels = file2matrix(filename)
    showdatas(datingDataMat, datingLabels)
    #classifyThings()
    button1(1)
    root.mainloop()         