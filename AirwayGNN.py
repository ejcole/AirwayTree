import os
import scipy
from scipy.sparse import coo_matrix
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
from collections import Counter
#from random_graph_classification import create_airway_graph

def getFileNames(dir='1',t ='hdr'):
   out =[]
   dir = "E:\SpiromicsData"
   tag = t
   for root, dirs, files in os.walk(dir, topdown=False):
            for name in files:
               if(name[-3:]==tag):
                  #print(os.path.join(root, name))
                  out.append(os.path.join(root, name))
            for name in dirs:
               if (name[-3:] == tag):
                  #print(os.path.join(root, name))
                  out.append(os.path.join(root, name))
                  #print(os.path.join(root, name))
   print('Extracted File Names')
   #print(out[-1])
   return out

def CreateCOOMatrix_all():
    count=0
    filelist = getFileNames(t='xml')
    print("loaded file name")
    maxBP = 0
    for f in filelist[0:1]:
        print(f)
        count+=1
        row =[]
        col =[]
        data = []
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if (child.tag == "Branchpoints"):
                for c in child:
                    if(int(c.attrib['id'])>maxBP):
                        maxBP=int(c.attrib['id'])
        print(maxBP)
        # COO=scipy.sparse.coo_matrix((maxBP,maxBP))
        # print(COO.toarray())
        for child in root:
            if (child.tag == "SegmentNames"):
                for c in child:
                    #print(c.attrib['startBpId'],c.attrib['endBpId'])
                    row.append(c.attrib['startBpId'])
                    col.append(c.attrib['endBpId'])
                    data.append(1)
        if count%50==0:
            print(count)
        COO = scipy.sparse.coo_matrix((data,(row, col)),shape=(maxBP,maxBP))
        print(COO.row)
        print(COO.toarray())
        print(COO.shape)

def CreateCOOMatrix(f='none'):
    maxBP = 0
    print(f)
    row = []
    col = []
    data = []
    tree = ET.parse(f)
    root = tree.getroot()
    for child in root:
        if(child.tag == "Branchpoints"):
            for c in child:
                if(int(c.attrib['id']) > maxBP):
                    maxBP = int(c.attrib['id'])
    #print(maxBP)
    # COO=scipy.sparse.coo_matrix((maxBP,maxBP))
    # print(COO.toarray())
    for child in root:
        if(child.tag == "SegmentNames"):
            for c in child:
    # print(c.attrib['startBpId'],c.attrib['endBpId'])
                row.append(int(c.attrib['startBpId']))
                col.append(int(c.attrib['endBpId']))
                data.append(1)
    coo=scipy.sparse.coo_matrix((data, (row, col)), shape=(maxBP, maxBP))
    print(coo.toarray()[198][:])
    print(coo.row)
    print(coo.col)
    temp =np.stack([coo.row,coo.col],axis =0)
    print(temp)
    return(coo)


def main():
   filelist = getFileNames(t='xml')
   x = CreateCOOMatrix(filelist[0])
   d = create_airway_graph(240,x)
   print(d.edge_index.shape)
   print(d.x.shape)
   print(d.y)

if __name__ == '__main__':
    main()