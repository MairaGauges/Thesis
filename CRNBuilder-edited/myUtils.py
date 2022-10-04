import numpy
import pyvista as pv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def readVTKSeries(path,caseName):
    """
    This function reads the VTK/caseName+'.vtm.series' file and returns the list of files and the corresponding time
    """
    import json

    # read file
    VTKPREFIX='/VTK/'
    with open(path+caseName+VTKPREFIX+caseName+'.vtm.series', 'r') as myfile:
        data=myfile.read()

    # parse file
    obj = json.loads(data)

    timeList = []
    filesList = []
    for i,item in enumerate(obj['files']):
        print(item)
        print(item['name'][:-4])
        timeList.append(item['time'])
        filesList.append(path+caseName+VTKPREFIX+item['name'][:-4]+'/internal.vtu')
    return timeList,filesList


def getUtxt(file):
    myTable = pd.read_csv(file, sep='\s+')
    UzExp = myTable.get('U_ax')
    UtExp = myTable.get('U_tan')
    UrExp = myTable.get('U_r')



    expCoord = myTable.get('r')


    return expCoord, UzExp, UtExp, UrExp

def getUSim(VTK, level, radius):

    a = [0, -radius, level]
    b = [0, radius, level]

    res = 100
    lineGrid = VTK.sample_over_line(a, b, resolution=res)
    lineCoord = np.linspace(-radius, radius, res + 1)

    try:
        Uy = lineGrid.get_array('Uy')
        Ux = lineGrid.get_array('Ux')
        Uz = lineGrid.get_array('Uz')
    except:
        Uy = lineGrid.get_array('UMeany')
        Ux = lineGrid.get_array('UMeanx')
        Uz = lineGrid.get_array('UMeanz')

    zeros = np.zeros_like(Ux)


    Ut = (Ux)
    Ur = Uy

    # Uz = lineGrid.get_array('UMeanz')

    return lineCoord, Uz, Ut, Ur

def plotUexp(axis, levels, dataSet):
    m = 0
    for data in dataSet:
        n = 0   # To increment at each level
        for z in levels:
            axis[n, 0].set_ylabel(str(levels[n]) + ' mm')

            # ------------------------------------Experimental values at each height
            fileExp = './../../ExpResults/B4/NonReacting/Data/Vel_'
            if z<0:
                fileExp = fileExp + 'below_exp_' + str(z) + '.dat'
            else:
                fileExp = fileExp + 'exp_' + str(z) + '.dat'

            expCoord, UzExp, UtExp, UrExp = getUExp(fileExp)
            axis[n, 0].plot(expCoord, UzExp, 'ro', markersize=1, label=dataSet[m])
            axis[n, 1].plot(expCoord, UrExp, 'ro', markersize=1, label=dataSet[m])
            axis[n, 2].plot(expCoord, UtExp, 'ro', markersize=1, label=dataSet[m])

            n = n+1
        m = m+1

def plotUtxt(axis, levels, path, dataSet):

    for setName in dataSet:
        n = 0   # To increment at each level
        for z in levels:
            axis[n, 0].set_ylabel(str(levels[n]) + ' mm')

            # ------------------------------------Experimental values at each height
            if setName == 'Exp':
                myColor = 'ro'
                if z < 0:
                    file = path + 'Vel_below_exp_' + str(z) + '.dat'
                else:
                    file = path + 'Vel_exp_' + str(z) + '.dat'

                expCoord, UzExp, UtExp, UrExp = getUtxt(file)
                axis[n, 0].plot(expCoord, UzExp, 'ro', markersize=1, label=setName)
                axis[n, 1].plot(expCoord, UrExp, 'ro', markersize=1, label=setName)
                axis[n, 2].plot(expCoord, UtExp, 'ro', markersize=1, label=setName)
            else:
                myColor = ''
                if z<0:
                    file = path + 'Vel_below_' + str(-z) + '.dat'
                else:
                    file = path + 'Vel_' + str(z) + '.dat'

                expCoord, UzExp, UtExp, UrExp = getUtxt(file)
                axis[n, 0].plot(expCoord, UzExp, label=setName)
                axis[n, 1].plot(expCoord, UrExp, label=setName)
                axis[n, 2].plot(expCoord, UtExp, label=setName)

            n = n+1



def plotUsim(axis, levels, dataSet, path, casenames, colors):
    m = 0   # For dataset
    r = 0.0675

    for data in dataSet:
        n = 0  # For each level
        tList, fList = readVTKSeries(path, casenames[m])
        myVtk = pv.read(fList[-1])

        for z in levels:
            zCoord, UzSim, UtSim, UrSim = getUSim(myVtk, z/1000, r)

            axis[n, 0].plot(zCoord, UzSim, label= data, color=colors[m])
            axis[n, 1].plot(zCoord, UrSim, label= data, color=colors[m])
            axis[n, 2].plot(zCoord, UtSim, label= data, color=colors[m])
            n = n+1

        m = m+1

def getUPrime2Mean(VTK, level, radius):

    a = [0, -radius, level]
    b = [0, radius, level]

    res = 100
    lineGrid = VTK.sample_over_line(a, b, resolution=res)
    lineCoord = np.linspace(-radius, radius, res + 1)



    #Umean = lineGrid.get_array('UPrime2Mean')
    Uxx = lineGrid.get_array('UPrime2Meanxx')
    Uxy = lineGrid.get_array('UPrime2Meanxy')
    Uxz = lineGrid.get_array('UPrime2Meanxz')
    Uyy = lineGrid.get_array('UPrime2Meanyy')
    Uyz = lineGrid.get_array('UPrime2Meanyz')
    Uzz = lineGrid.get_array('UPrime2Meanzz')


    # Uz = lineGrid.get_array('UMeanz')

    return lineCoord, Uxx, Uxy, Uxz, Uyy, Uyz, Uzz

def plotUPrime2Mean(axis, levels, dataSet, path, casenames):
    m = 0   # For dataset
    r = 0.0675

    for data in dataSet:
        n = 0  # For each level
        tList, fList = readVTKSeries(path, casenames[m])
        myVtk = pv.read(fList[-1])
        for z in levels:
            zCoord, Uxx, Uxy, Uxz, Uyy, Uyz, Uzz = getUPrime2Mean(myVtk, z/1000, r)



            # Uxrms = np.sqrt(Uxx)
            # Uyrms = np.sqrt(Uyy)
            # Uzrms = np.sqrt(Uzz)

            # Uxrms = np.sqrt(np.sqrt(Uxx * Uxx + Uxy * Uxy + Uxz * Uxz))       #This calculation is not correct. Use only diagonal elements of the tensor as below!!
            # Uyrms = np.sqrt(np.sqrt(Uxy * Uxy + Uyy * Uyy + Uyz * Uyz))
            # Uzrms = np.sqrt(np.sqrt(Uxz * Uxz + Uyz * Uyz + Uzz * Uzz))

            Uxrms = np.sqrt(Uxx)
            Uyrms = np.sqrt(Uyy)
            Uzrms = np.sqrt(Uzz)

            axis[n, 0].plot(zCoord, Uzrms, label= dataSet[m])
            axis[n, 1].plot(zCoord, Uyrms, label= dataSet[m])
            axis[n, 2].plot(zCoord, Uxrms, label= dataSet[m])




            # axis[n, 0].plot(zCoord, Uzrms)
            # axis[n, 1].plot(zCoord, Uyrms)
            # axis[n, 2].plot(zCoord, Uxrms)

            n = n+1
        m = m+1




def getUrmsTxt(file):
    myTable = pd.read_csv(file, sep='\s+')
    UzExp = myTable.get('U_ax_rms')
    UtExp = myTable.get('U_tan_rms')
    UrExp = myTable.get('U_r_rms')



    expCoord = myTable.get('r')


    return expCoord, UzExp, UtExp, UrExp

def plotUrmstxt(axis, levels, path, dataSet):

        for setName in dataSet:
            n = 0  # To increment at each level
            for z in levels:
                axis[n, 0].set_ylabel(str(levels[n]) + ' mm')

                # ------------------------------------Experimental values at each height
                if setName == 'Exp':
                    if z < 0:
                        file = path + 'Vel_below_exp_' + str(z) + '.dat'
                    else:
                        file = path + 'Vel_exp_' + str(z) + '.dat'

                    expCoord, UzExp, UtExp, UrExp = getUrmsTxt(file)
                    axis[n, 0].plot(expCoord, UzExp, 'ro', markersize=1, label=setName)
                    axis[n, 1].plot(expCoord, UrExp, 'ro', markersize=1, label=setName)
                    axis[n, 2].plot(expCoord, UtExp, 'ro', markersize=1, label=setName)
                else:
                    if z < 0:
                        file = path + 'Vel_below_' + str(-z) + '.dat'
                    else:
                        file = path + 'Vel_' + str(z) + '.dat'

                    expCoord, UzExp, UtExp, UrExp = getUrmsTxt(file)
                    axis[n, 0].plot(expCoord, UzExp, label=setName)
                    axis[n, 1].plot(expCoord, UrExp, label=setName)
                    axis[n, 2].plot(expCoord, UtExp, label=setName)

                n = n + 1

                
def getSimVarsOnLevels(VTK, levels, radius, res, variables):
    results = []
    for z in levels:
        a = [-radius, 0, z/1000]
        b = [radius, 0, z/1000]
        lineGrid = VTK.sample_over_line(a, b, resolution=res)
        lineCoord = np.linspace(-radius, radius, res + 1)
        for var in variables:
            results.append(lineGrid.get_array(var))
    return results

def getSimVarAveragedOverRadius(VTK, levels, radius, res, variables):
    results = []
    for z in levels:
        lineCoord = np.linspace(0, radius, res + 1)
        for var in variables:
            for r in lineCoord:
                center = [0, 0, z/1000]
                normal = [0, 0, 1]
                polar = [0, r, 0]
                angle = 360
                circGrid = VTK.sample_over_circular_arc_normal(center, normal=normal,polar=polar, angle=angle, resolution=20)
                var_array = circGrid.get_array(var)
                val = (np.sum(var_array))/var_array.size

                results.append(val)
    return results
