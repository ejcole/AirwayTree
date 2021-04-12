import os
import SimpleITK as ITK
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import _pickle as pickle
from collections import Counter

def xmlCheck(f = "E:\SpiromicsData\Export_1\H-14361_WF120165_SPI-WF120165-V5_INSPIRATION 0.625 STANDARD\ZUNU_vida-xmlTree.xml"):
   tree = ET.parse(f)
   root = tree.getroot()
   branchp = {}
   segName = {}
   for child in root:
      print(child.tag, child.attrib)
      if(child.tag == "Branchpoints"):
         for c in child:
            #print(c.tag,c.attrib)
            branchp[c.attrib['id']] = (c.attrib['x'],c.attrib['y'],c.attrib['z'])
      if (child.tag == "SegmentNames"):
         for c in child:
            segName[c.attrib['anatomicalName']] = (c.attrib['startBpId'],c.attrib['endBpId'],c.attrib['linkIds'])
   print(segName.keys())
   #print(branchp.keys())
   total=0
   for key in branchp.keys():
      total = total + int(key)
   print(total)
   # for i in range(241):
   #    print(i)
   check = list(range(240))
   # print(check)
   # print(len(check))
   a = branchp.keys()
   diff = []
   for i in a:
      diff.append(i)
   x = sorted(diff, key=int)
   x = [int(i) for i in x]
   print(len(x))
   print(list(set(check) - set(x)))
   # print(x)
   #print(a)
   # print(sanity)
   # print(sanity == total)
   return branchp

  #print(children)




def getSegmentnames():
   count = 0
   segName = {}
   filelist = getFileNames(t='xml')
   print("loaded file names")
   for f in filelist:
      tree = ET.parse(f)
      root = tree.getroot()
      for child in root:
         if (child.tag == "SegmentNames"):
            for c in child:
               segName[c.attrib['anatomicalName']] = segName.get(c.attrib['anatomicalName'],0) + 1
      count = count + 1
      if count%50 ==0:
         print(count)

   with open('Segments.txt', 'w') as f:
      print(segName, file=f)
   return segName

def count():
   img = ITK.ReadImage("E:\SpiromicsData\Export_1\H-14361_WF120165_SPI-WF120165-V5_INSPIRATION 0.625 STANDARD/ZUNU_vida-aircolor.hdr")
   imgAr = ITK.GetArrayViewFromImage(img)
   print(imgAr.shape)
   x,y,z = imgAr.shape
   # print(x,y,z)
   # print(np.unique(imgAr))
   print(len(np.unique(imgAr)))
   flat = imgAr.flatten()
   n, bins, patches = plt.hist(flat,bins=flat.max()-1,range= (1,flat.max()))
   # print(flat.max())
   # print(n)
   #print(n==10)
   plt.yscale('log')
   plt.xlim(1,245)
   plt.title("Histogram of Branch Voxels")
   plt.ylabel("Log of Voxel Count")
   plt.xlabel("Branch Number")
   plt.show()

def airwayVoxel(f='1'):
   img = ITK.ReadImage(f)
   imgAr = ITK.GetArrayViewFromImage(img)
   print(imgAr.shape)
   flat = imgAr.flatten()
   flat = np.array(flat)
   count = 0
   print("Begin Count")
   #count = sum(i > 0 for i in flat)
   # for i in flat:
   #    if i>0:
   #       count = count +1
   #       if i%10000==0:
   #          print(i)
   count = np.count_nonzero(flat)
   print("countFinished")
   return count

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

def getBranchNumber(f):
   img = ITK.ReadImage(f)
   imgAr = ITK.GetArrayViewFromImage(img)
   # print(imgAr.shape)
   # x, y, z = imgAr.shape
   # print(x, y, z)
   # print(np.unique(imgAr))
   return (len(np.unique(imgAr)))
   # flat = imgAr.flatten()
   # n, bins, patches = plt.hist(flat, bins=flat.max() - 1, range=(1, flat.max()))
   # # print(flat.max())
   # # print(n)
   # # print(n==10)
   # plt.yscale('log')
   # plt.xlim(1, 245)
   # plt.title("Histogram of Branch Voxels")
   # plt.ylabel("Log of Voxel Count")
   # plt.xlabel("Branch Number")
   # plt.show()

def branch_inventory():
   filelist = getFileNames()
   branchNumbers = []
   files_processed = 0
   for x in filelist:
      print(x)
      branchNumbers.append(branchNumbers(x))
      if (files_processed % 25 == 0):
         print(files_processed)
      files_processed = files_processed + 1
   with open("BranchNumbers.txt", "w") as txt_file:
      for fname, num in zip(filelist, branchNumbers):
         txt_file.write(str(num) + " " + fname + "\n")  # works with any number of elements in a line

   TreeSize = []
   files_processed = 0
   for x in filelist:
      print(x)
      TreeSize.append(airwayVoxel(x))
      if (files_processed % 25 == 0):
         print(files_processed)
      files_processed = files_processed + 1
   with open("AirwaySize.txt", "w") as txt_file:
      for fname, num in zip(filelist, TreeSize):
         txt_file.write(str(num) + " " + fname + "\n")  # works with any number of elements in a line

def createPlots():
   a_file = open("AirwaySize_v1.txt")
   lines = a_file.readlines()
   a_file.close()
   voxels = []
   for line in lines:
      voxels.append(int(line[0:6]))
   plt.hist(voxels, bins=200)
   plt.title('voxels per image')
   plt.show()
   a_file = open("Bnum_v1.txt")
   lines = a_file.readlines()
   a_file.close()
   branches = []
   for line in lines:
     branches.append(int(line[0:3]))
   plt.hist(branches, bins=200)
   plt.title("branches per image")
   plt.show()

def getLeftRightSegements():
   dicts_from_file = {}
   with open('Segments_v1.txt', 'r') as inf:
      for line in inf:
         dicts_from_file=(eval(line))
   keys = dicts_from_file.keys()
   values = dicts_from_file.values()
   print(list(keys))
   left = []
   right = []
   unsorted = []
   for k in keys:
      if (k[0] == 'r') or (k[0] == 'R'):
         right.append(k)
      elif (k[0] == 'l') or (k[0] == 'L'):
         left.append(k)
      else:
         unsorted.append(k)
   # with open('left.txt', 'w') as filehandle:
   #    for listitem in left:
   #       filehandle.write('%s\n' % listitem)
   # with open('right.txt', 'w') as filehandle:
   #    for listitem in right:
   #       filehandle.write('%s\n' % listitem)

def createList(file):
    l = []
    with open(file, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            currentPlace = line[:-1]
            # add item to the list
            l.append(currentPlace)
    return l

def leftrightvoxelcount():
    left = createList('left.txt')
    right = createList('right.txt')
    filelist_hdr = getFileNames(t='hdr')
    filelist_xml = getFileNames(t='xml')
    results =[]
    for t,im in zip(filelist_xml,filelist_hdr):
        print(t)

        if(t[30:40] == im[30:40]):
            tree = ET.parse(t)
            root = tree.getroot()
            branchp = {}
            segName = {}
            for child in root:
                #print(child.tag, child.attrib)
                if (child.tag == "Branchpoints"):
                    for c in child:
                        #print(c.tag,c.attrib)
                        branchp[c.attrib['id']] = (c.attrib['x'], c.attrib['y'], c.attrib['z'])
                if (child.tag == "SegmentNames"):
                    for c in child:
                        segName[c.attrib['anatomicalName']] = (c.attrib['startBpId'], c.attrib['endBpId'], c.attrib['linkIds'])
            #print(segName.keys())
            temp = []
            level_dict = {}
            for k in segName.keys():
                #print(str(k) +" : " + str(segName[k][2]))
                temp.append(segName[k][2])
                key = segName[k][2]
                if key not in level_dict:
                    level_dict[key] = k
            #print(sorted(temp,key=int))

            img = ITK.ReadImage(im)
            imgAr = ITK.GetArrayViewFromImage(img)
            #print(imgAr.shape)
            x,y,z = imgAr.shape
            total_image = x*y*z
            flat = imgAr.flatten()
            flat = np.array(flat)
            index =[]
            count = []
            #print("Begin Count")
            left_total = 0
            right_total = 0
            leftover = 0
            bg = 0

            for j in np.unique(flat):
                index.append(j)
                count.append(np.count_nonzero(flat == j))
           # print(index,count)
            #print(level_dict.keys())
           # print(sum(count))
            print("startingCount")
            for i,j in zip(index,count):
                #print(i,j)
                i = str(i)
                if str(i) in level_dict.keys():
                    seg = level_dict[i]
                    #print(seg)
                    if seg in left:
                        left_total += j
                    elif seg in right:
                        right_total += j
                    else:
                        leftover += j
                else:
                    bg += j
            results.append((left_total,right_total,leftover,bg,total_image))
           # print("Count Done")
           # print("Left  Right")
           # print(left_total,right_total)
           # print("Total Lung")
           # print(left_total+right_total)
           # print("Unsorted")
           # if leftover != 0:
           #    print(leftover)
           # print("Background")
           # print(bg)
           # print("Image Total")
           # print(left_total+right_total+leftover+bg)
        else:
           print("File Error")
           for i,j in zip (im,t):
               if i != j:
                   print(i,j)
           print(im)
           print(t)
    print(results)
    # results = [(0,1,2,3,4),(5,6,7,8,9)]
    # filelist_xml=['test1','test2']
    # filelist_hdr=['test3','test4']
    # with open('BranchSizes.txt', 'w') as f:
    #     f.write("Left   Right  Leftover  Background    Total" + "\n")
    #     for t, x, h in zip(results, filelist_hdr, filelist_xml):
    #         print(t)
    #         f.write('%s  %s  %s    %s     %s ' % t)
    #         f.write(x + '\n')

def createHistBranchsize():
    left = []
    right = []
    with open('BranchSizes.txt') as f:
        next(f)
        for line in f:
            t = line.split()
            left.append(float(t[0]))
            right.append(float(t[1]))
    #Create plot for left trees
    plt.hist(left,bins=75)
    plt.title('Left1')
    plt.xlabel("Voxels")
    plt.ylabel('Trees')
    plt.show()
    #Create plot for right trees
    plt.hist(right,bins=75)
    plt.xlabel("Voxels")
    plt.ylabel('Trees')
    plt.title('Right1')
    plt.show()
    relative =[]
    for i,j in zip(left,right):
        relative.append(i/j)
    plt.hist(relative,bins=75)
    plt.xlabel("Voxel Ratio")
    plt.ylabel('Trees')
    plt.title('Left/Right1')
    plt.show()

def createHist():
    dicts_from_file = {}
    segNames = []
    segValues = []

    with open('Segments_v1.txt', 'r') as inf:
        for line in inf:
            dicts_from_file = (eval(line))
    keys = dicts_from_file.keys()

    left = createList('left.txt')
    right = createList('right.txt')

    for i in left:
        print(i, dicts_from_file[i])
        segNames.append(i)
        segValues.append(dicts_from_file[i])

    for j in right:
        print(j, dicts_from_file[j])
        segNames.append(j)
        segValues.append(dicts_from_file[j])
    fig = plt.figure()
    #ax = fig.add_axes([0,0,1,1])
    print(segNames)
    print(segValues)
    print(min(segValues))
    segValues_norm =[]
    for i in segValues:
        segValues_norm.append(i-477)
    plt.bar(segNames,segValues_norm)
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.title("Segements - 477")
    plt.show()
    for i,j in zip(segValues_norm,segNames):
        if int(i) < 70:
            print(i,j)
    createHistBranchsize()

def voxelSize():
    count =0
    voxelsize = 0
    voxelList = []
    filelist = getFileNames(t='xml')
    print("loaded file names")
    for f in filelist:
        count = count +1
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
            if (child.tag == "VoxelDimensions"):
                #print(child.attrib['x'])
                voxelsize = float(child.attrib['x'])*float(child.attrib['y'])*float(child.attrib['z'])
                voxelList.append(voxelsize)
        if count%50==0:
            print(count)


    with open('VoxelSizes.txt', 'w') as f:
        f. write("VoxelSize       Filename" + "\n")
        for i,j in zip(voxelList,filelist):
            f.write('%s  ' % i)
            f.write(j + '\n')

def Voxelsize():
    vSize = []
    fileNameCheck =[]
    fileNameCheck1 =[]
    file = 'VoxelSizes.txt'
    left =[]
    right = []
    leftover = []
    background = []
    total = []
    place =0
    with open(file, 'r') as filehandle:
        next(filehandle)
        for line in filehandle:
            # remove linebreak which is the last character of the string
            #currentPlace = line[:-1]
            split = line.split()
            vSize.append(split[0])
            fileNameCheck.append(split[1])
    file ='BranchSizes.txt'
    results=[]
    with open(file, 'r') as filehandle:
        next(filehandle)
        for line in filehandle:
            split = line.split()
            #print(split[0],vSize[place])
            left.append((float(split[0])*float(vSize[place]))/1000)
            right.append((float(split[1]) * float(vSize[place]))/1000)
            leftover.append((float(split[2]) * float(vSize[place]))/1000)
            background.append((float(split[3]) * float(vSize[place]))/1000)
            total.append((float(split[4])*float(vSize[place]))/1000)
            fileNameCheck1.append(split[5])
            place+=1
    for i,j,k,l,m in zip(left, right,leftover,background,total):
        results.append((i,j,k,l,m))

    with open('BranchSizes_VoxelAdjusted.txt', 'w') as f:
        f.write("Left                        Right                          Leftover                           Background                 Total" + "\n")
        # for i,j,k,l,m,n,o in zip(left, right,leftover,background,total, fileNameCheck,fileNameCheck1):
        #     f.write('%s  %s  %s    %s     %s      %s  %s ' % i,j,k,l,m,n,o)
        #     f.write('\n')
        for t, x in zip(results, fileNameCheck):
            f.write('%s           %s              %s              %s              %s ' % t)
            f.write(x + '\n')

    for i,j in zip(fileNameCheck,fileNameCheck1):
        if i!=j:
            print(i,j)

# def main():
    # data = []
    # left = []
    # right = []
    # leftover = []
    # background = []
    # total = []
    # fileName = []
    # file = 'BranchSizes_VoxelAdjusted.txt'
    # with open(file, 'r') as filehandle:
    #     next(filehandle)
    #     for line in filehandle:
    #         floaters = []
    #         # remove linebreak which is the last character of the string
    #         # currentPlace = line[:-1]
    #         split = line.split()
    #         left.append(float(split[0]))
    #         right.append(float(split[1]))
    #         leftover.append(float(split[2]))
    #         background.append(float(split[3]))
    #         total.append(float(split[4]))
    #         fileName.append(split[5])
    # plt.title("Left")
    # plt.hist(left,bins=75)
    # plt.xlabel("CC")
    # plt.show()
    # plt.title("Right")
    # plt.hist(right,bins=75)
    # plt.xlabel("CC")
    # plt.show()
    # plt.title("Trachea")
    # plt.xlabel("CC")
    # plt.hist(leftover, bins=75)
    # plt.show()
    # relative =[]
    # for i,j in zip(left,right):
    #     relative.append(i/j)
    # plt.hist(relative,bins=75)
    # plt.title('Left/Right')
    # plt.xlabel("CC")
    # plt.show()
    # createHist()
    # voxelSize()
def main():
    Path1 = ['Trachea','RMB','RUL','RB1','rb1_b']
    Path2 = ['Trachea','RMB','BronInt','RB4+5','RB4','rb4_b']
    Path3 = ['Trachea','RMB','BronInt','RLL7','RLL','RB10','rb10_b']
    Path4 = ['Trachea','LMB','lb1_b','LUL','LB1+2','LB1']
    Path5 = ['Trachea','LMB','LUL','LB4+5','LB4']
    Path6 = ['Trachea','LMB','LLB','LB10','lb10_b']
    count = 0
    pathways =[]
    filelist = getFileNames(t='xml')
    print("loaded file names")
    path_comp = [0, 0, 0, 0, 0, 0]
    #missing = [[],[],[],[],[],[]]
    for f in filelist:
        missing = [[], [], [], [], [], []]
        segName = []
        tree = ET.parse(f)
        root = tree.getroot()
        for child in root:
         if (child.tag == "SegmentNames"):
            for c in child:
               segName.append(c.attrib['anatomicalName'])

        check1 = all(item in segName for item in Path1)
        if check1==False:
            for i in Path1:
                if i not in segName:
                    missing[0].append(i)
        # if(len(missing[0])>0):
        #     print(missing[0], f)

        check2 = all(item in segName for item in Path2)
        if check2==False:
            for i in Path2:
                if i not in segName:
                    missing[1].append(i)
        # if (len(missing[1]) > 0):
        #     print(missing[1], f)
        check3 = all(item in segName for item in Path3)
        if check3==False:
            for i in Path3:
                if i not in segName:
                    missing[2].append(i)
        # if (len(missing[2]) > 0):
        #     print(missing[2], f)
        check4 = all(item in segName for item in Path4)
        if check4==False:
            for i in Path4:
                if i not in segName:
                    missing[3].append(i)
        # if (len(missing[3]) > 0):
        #     print(missing[3], f)
        check5 = all(item in segName for item in Path5)
        if check5==False:
            for i in Path5:
                if i not in segName:
                    missing[4].append(i)
        # if (len(missing[4]) > 0):
        #     print(missing[4], f)
        check6 = all(item in segName for item in Path6)
        if check6==False:
            for i in Path6:
                if i not in segName:
                    missing[5].append(i)
        # if (len(missing[5]) > 0):
        #     print(missing[5], f)
        allCheck =(check1,check2,check3,check4,check5,check6)
        pathways.append(allCheck)
        flag=0
        accum = []
        for i in range(0,6):
            #print(allCheck[i])
            if (len(missing[i])>0):
                accum.append(missing[i])
                flag=1

            if allCheck[i] == True:
                path_comp[i] = path_comp[i] + 1
        if flag == 1:
            print(f)
            print(accum)

        count = count + 1
        if count%20 ==0:
            print(count)
    for j in missing:
        print(Counter(j))

    print(path_comp)
    with open('SarpPath_Missing.txt', 'w') as f:
        f.write("Pathways\t\t\t\t\t\t\t FileName" + "\n")
        for i, j in zip(filelist,pathways):
            f.write("%s %s %s %s %s %s " %j)
            f.write(i+'\n')

if __name__ == '__main__':
    main()